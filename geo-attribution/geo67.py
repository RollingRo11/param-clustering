"""Training-free parameter decomposition from attribution geometry — pile-4L 67M.

Pipeline (each stage a subcommand; collect/gram are torchrun-able across 2 GPUs):
  collect : K forward+backward passes per batch at scaled weights (IG sensor),
            cache per-module (pre, gpost) at sampled token positions. DDP-sharded.
  gram    : exact signed attribution gram G_ij = sum_W sum_kk' (g_i*g_j)^T W^2 (p_i*p_j)
            via the pairwise identity — v(x) never materialized. Modules sharded
            across ranks, allreduce-summed.
  factor  : double-center + cosine-normalize G, eigendecompose, pick C (eigengap or
            --C), spherical k-means on the spectral embedding -> cluster labels.
  extract : per cluster per module, anchor top-m singular directions on the
            residual-stream side; joint ridge solve for free factors; explicit
            residual rho = W - sum(components). Faithful by construction.
            Saves banks.pt in ComponentBank layout (A [C,d_in,m], B [C,m,d_out]).
  eval    : held-out Pile: IG attribution gates (threshold), full-gate KL/token,
            keep-top-j KL curve, gates/token, residual-only inertness control,
            zeroed-matrices off baseline, faithfulness identity check.
  canary  : synthetic induction referee: copy accuracy under gating, dgate-contrast
            component ranking, top-k ablation vs random-k nulls.

Run from anywhere; imports /workspace/param-decomp for target + data plumbing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.environ.get("PARAM_DECOMP_ROOT", "/workspace/param-decomp"))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

OUT_ROOT = Path("/workspace/circuit-decomp/geo-attribution/out")

MODULES: list[str] = []          # filled from pile_4L at load time
# modules whose OUTPUT lives in the residual stream -> anchor on the g side;
# all others read the residual stream -> anchor on the p side.
WRITE_SIDE_SUFFIX = ("attn.o_proj", "mlp.down_proj")


def log(msg: str):
    r = int(os.environ.get("RANK", 0))
    print(f"[r{r}] {msg}", flush=True)


def ddp_setup():
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}" if ddp else "cuda"
    if ddp:
        # Establish the CUDA mapping before NCCL is initialized so barriers do
        # not have to guess which device belongs to this process.
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(device))
    return ddp, rank, world, device


LOCAL_TARGET = str(OUT_ROOT / "target_local" / "ckpt" / "model_step_99999.pt")


def load_target(device: str) -> nn.Module:
    from nano_param_decomp.pile_4L import C_PER_MODULE_4L, load_paper_target_model
    MODULES[:] = list(C_PER_MODULE_4L.keys())
    # local-path branch of PretrainRunInfo.from_path — avoids wandb auth entirely
    target = load_paper_target_model(run_path=LOCAL_TARGET).float().to(device)
    for p in target.parameters():
        p.requires_grad_(True)   # activation grads; never optimized
    return target


def is_write_side(path: str) -> bool:
    return path.endswith(WRITE_SIDE_SUFFIX)


# ------------------------------------------------------------- GIM sensor ----
# GIM (arXiv 2505.17630): corrected backprop for attribution through the
# modules where plain gradients mis-credit — temperature-adjusted softmax
# jacobian (anti-saturation; credit reaches the QK path) and frozen
# normalization statistics in RMSNorm. Forward values are UNCHANGED (identical
# model); only the backward differs, so faithfulness and gated behavior are
# untouched — this is purely a sensor swap. Composes with the IG weight-scaling
# path (K passes). Not implemented: gradient normalization at multiplicative
# interactions (no gated MLP in this target; q·k and att@v products left as-is).

class _GimSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, tau):
        p = torch.softmax(x, dim=dim)
        pt = torch.softmax(x / tau, dim=dim)
        ctx.save_for_backward(pt)
        ctx.dim = dim
        return p
    @staticmethod
    def backward(ctx, g):
        (pt,) = ctx.saved_tensors
        gx = (g - (g * pt).sum(ctx.dim, keepdim=True)) * pt
        return gx, None, None


def apply_gim(target: nn.Module, tau: float = 2.0):
    """Idempotent: patch RMSNorms (freeze rsqrt), force manual attention
    (exposes F.softmax), and swap F.softmax for the GIM version on 4D inputs
    (attention scores only; CE/log_softmax paths untouched)."""
    if getattr(target, "_gim_applied", False):
        return
    target._gim_applied = True
    for mod in target.modules():
        if type(mod).__name__ == "LlamaRMSNorm":
            def rms_fwd(hidden_states, _m=mod):
                dt = hidden_states.dtype
                h = hidden_states.to(torch.float32)
                var = h.pow(2).mean(-1, keepdim=True)
                h = h * torch.rsqrt(var + _m.variance_epsilon).detach()
                return _m.weight * h.to(dt)
            mod.forward = rms_fwd
        if hasattr(mod, "flash_attention") and mod.flash_attention:
            mod.flash_attention = False
            if not hasattr(mod, "bias"):
                dev = next(mod.parameters()).device
                mod.register_buffer("bias", torch.tril(
                    torch.ones(1024, 1024, device=dev)).view(1, 1, 1024, 1024),
                    persistent=False)
    orig_softmax = F.softmax
    def gim_softmax(input, dim=None, *a, **kw):  # noqa: A002 — match F.softmax
        if input.dim() == 4 and not kw.get("dtype"):
            return _GimSoftmax.apply(input, dim, tau)
        return orig_softmax(input, dim=dim, *a, **kw)
    F.softmax = gim_softmax
    log(f"GIM sensor active (tau={tau}): manual attention, patched softmax "
        f"backward, frozen RMSNorm stats")


class Capture:
    """Monkeypatch decomposed linears: weight scaling for the IG path + pre cache.

    ``linear_kernel`` is injectable so the fast collector can compile the pure
    linear operation while leaving cache mutation in eager Python. Compiling
    the cache-writing closure itself disconnects its side-effect-only tensors
    from AOTAutograd; keeping mutation outside the compiled boundary makes the
    captured outputs ordinary autograd graph nodes.
    """

    def __init__(self, target: nn.Module, linear_kernel: Callable = F.linear):
        self.target = target
        self.wscale = 1.0
        self.cache: dict[str, dict] = {}
        self.linear_kernel = linear_kernel
        for path in MODULES:
            lin = target.get_submodule(path)
            lin._path = path
            lin._orig_forward = lin.forward
            lin.forward = self._mk(lin)

    def _mk(self, lin: nn.Linear):
        def fwd(x: Tensor) -> Tensor:
            w = lin.weight if self.wscale == 1.0 else lin.weight * self.wscale
            out = self.linear_kernel(x, w, lin.bias)
            self.cache[lin._path] = {"pre": x, "post": out}
            return out
        return fwd

    def run(self, idx: Tensor):
        self.cache = {}
        logits = self.target(idx)
        return logits, self.cache


def ce_sum(logits: Tensor, idx: Tensor) -> Tensor:
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), idx[:, 1:].flatten(),
                           reduction="sum")


def scalar_sum(logits: Tensor, idx: Tensor, mode: str = "ce") -> Tensor:
    """Attribution scalar. ce: summed CE vs ground truth (default).
    logp_pred: summed log-prob of the model's own argmax token ("what the
    model says" rather than "its error"; differs from ce exactly on wrong
    predictions; both saturate on confident tokens). logit_pred: raw logit of
    the argmax token (saturation-free direct-logit attribution)."""
    if mode == "ce":
        return ce_sum(logits, idx)
    if mode == "equal_reward":
        # The streaming collector uses this GIM objective: every observed
        # next-token logit receives the same unit output cotangent. Keeping the
        # implementation here as well ensures reusable banks are ranked with
        # the same scalar they were collected with.
        return logits[:, :-1].float().gather(-1, idx[:, 1:, None]).sum()
    if mode not in {"logp_pred", "logit_pred"}:
        raise ValueError(f"unknown attribution scalar: {mode}")
    pred = logits[:, :-1].argmax(-1).detach()
    z = (F.log_softmax(logits[:, :-1].float(), -1) if mode == "logp_pred"
         else logits[:, :-1].float())
    return z.gather(-1, pred[..., None]).sum()


# ----------------------------------------------------------------- collect ----

def stage_collect(args):
    ddp, rank, world, device = ddp_setup()
    target = load_target(device)
    if args.sensor == "gim":
        apply_gim(target, args.gim_tau)
    from nano_param_decomp.pile_4L import make_loader
    loader = make_loader(args.batch_seqs * world, args.seq_len, rank, world,
                         "train", args.seed)
    cap = Capture(target)
    n_local = args.n_positions // world
    S = args.pos_per_seq
    n_batches = math.ceil(n_local / (args.batch_seqs * S))
    gen = torch.Generator().manual_seed(args.seed + 7 * rank)

    store = {p: {"p": [[] for _ in range(args.ig_k)],
                 "g": [[] for _ in range(args.ig_k)]} for p in MODULES}
    toks, nexts = [], []
    t0 = time.time()
    for b in range(n_batches):
        idx = next(loader).to(device)
        B, T = idx.shape
        # positions sampled once per batch, shared across the K IG passes;
        # exclude the tail (little/no future loss -> tiny gpost) and the first tokens
        pos = torch.stack([torch.randperm(T - 6, generator=gen)[:S] + 4
                           for _ in range(B)]).to(device)          # [B, S]
        bi = torch.arange(B, device=device)[:, None].expand(-1, S)
        for k in range(1, args.ig_k + 1):
            cap.wscale = k / args.ig_k
            target.zero_grad(set_to_none=True)
            logits, cache = cap.run(idx)
            s = scalar_sum(logits, idx, args.scalar)
            posts = [cache[p]["post"] for p in MODULES]
            gposts = torch.autograd.grad(s, posts)
            for p, g in zip(MODULES, gposts, strict=True):
                pre = cache[p]["pre"].detach()
                store[p]["p"][k - 1].append(pre[bi, pos].reshape(-1, pre.shape[-1]).bfloat16().cpu())
                store[p]["g"][k - 1].append(g[bi, pos].reshape(-1, g.shape[-1]).bfloat16().cpu())
        toks.append(idx[bi, pos].reshape(-1).cpu())
        nexts.append(idx[bi, (pos + 1).clamp(max=T - 1)].reshape(-1).cpu())
        if b % 8 == 0:
            log(f"collect batch {b}/{n_batches} ({time.time()-t0:.0f}s)")

    out = {"modules": MODULES, "ig_k": args.ig_k, "n": None,
           "sensor": args.sensor, "gim_tau": args.gim_tau, "scalar": args.scalar,
           "tok": torch.cat(toks)[:n_local], "next": torch.cat(nexts)[:n_local]}
    out["n"] = out["tok"].shape[0]
    for p in MODULES:
        out[p] = {
            "p": torch.stack([torch.cat(store[p]["p"][k])[:n_local] for k in range(args.ig_k)]),
            "g": torch.stack([torch.cat(store[p]["g"][k])[:n_local] for k in range(args.ig_k)]),
        }  # each [K, n_local, d]
    args.dir.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dir / f"collect_rank{rank}.pt")
    log(f"collect done: {out['n']} positions -> collect_rank{rank}.pt "
        f"({time.time()-t0:.0f}s)")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def load_collect(dirp: Path) -> dict:
    shards = sorted(dirp.glob("collect_rank*.pt"))
    assert shards, f"no collect shards in {dirp}"
    parts = [torch.load(s, weights_only=True, map_location="cpu") for s in shards]
    out = {"modules": parts[0]["modules"], "ig_k": parts[0]["ig_k"],
           "sensor": parts[0].get("sensor", "ig"), "scalar": parts[0].get("scalar", "ce"),
           "gim_tau": parts[0].get("gim_tau", 2.0)}
    out["tok"] = torch.cat([q["tok"] for q in parts])
    out["next"] = torch.cat([q["next"] for q in parts])
    for p in parts[0]["modules"]:
        out[p] = {"p": torch.cat([q[p]["p"] for q in parts], dim=1),
                  "g": torch.cat([q[p]["g"] for q in parts], dim=1)}
    out["n"] = out["tok"].shape[0]
    return out


def load_extract_collect(dirp: Path) -> dict:
    """Load extraction metadata, leaving streamed module tensors on disk."""
    manifest_path = dirp / "stream_collection.json"
    if not manifest_path.exists():
        return load_collect(dirp)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "module_shards_v1":
        raise ValueError(f"unsupported collection format in {manifest_path}")
    parts = [torch.load(dirp / f"stream_meta_rank{rank}.pt",
                        weights_only=True, map_location="cpu")
             for rank in range(int(manifest["world"]))]
    if any(not part.get("complete") for part in parts):
        raise ValueError(f"stream collection in {dirp} has incomplete rank metadata")
    valid = [part["labels"] >= 0 for part in parts]
    out = {
        "modules": manifest["modules"], "ig_k": int(manifest["ig_k"]),
        "sensor": manifest.get("sensor", "ig"),
        "scalar": manifest.get("scalar", "ce"),
        "gim_tau": manifest.get("gim_tau", 2.0),
        "tok": torch.cat([p["tok"][m] for p, m in zip(parts, valid)]),
        "next": torch.cat([p["next"][m] for p, m in zip(parts, valid)]),
        "_stream_manifest": manifest, "_stream_parts": parts,
        "_stream_dir": dirp,
    }
    out["n"] = out["tok"].shape[0]
    return out


def load_extract_module(col: dict, path: str) -> dict[str, Tensor]:
    """Load one module from BF16 stream shards; legacy collections pass through."""
    if "_stream_manifest" not in col:
        return col[path]
    manifest = col["_stream_manifest"]
    module_index = manifest["modules"].index(path)
    pieces = {"p": [], "g": []}
    for rank, part in enumerate(col["_stream_parts"]):
        quota = int(part["quota"])
        capacity = int(manifest["C"]) * quota
        if part["labels"].numel() != capacity:
            raise ValueError(f"rank {rank} reservoir metadata has the wrong capacity")
        slots = (part["labels"] >= 0).nonzero(as_tuple=False).flatten().numpy()
        if not len(slots):
            continue
        for kind in ("p", "g"):
            dim = int(manifest["dims"][path][kind])
            file = (col["_stream_dir"] / f"reservoir_rank{rank}"
                    / f"module_{module_index:03d}_{kind}.bf16")
            expected = int(manifest["ig_k"]) * capacity * dim * 2
            if not file.exists() or file.stat().st_size != expected:
                raise ValueError(f"invalid streamed module shard {file}")
            mm = np.memmap(file, dtype=np.uint16, mode="r",
                           shape=(int(manifest["ig_k"]), capacity, dim))
            bits = torch.from_numpy(np.ascontiguousarray(mm[:, slots]))
            pieces[kind].append(bits.view(torch.bfloat16))
    if not pieces["p"]:
        raise ValueError(f"stream reservoir contains no examples for {path}")
    return {kind: torch.cat(values, dim=1) for kind, values in pieces.items()}


# -------------------------------------------------------------------- gram ----

def stage_gram(args):
    ddp, rank, world, device = ddp_setup()
    target = load_target(device)
    col = load_collect(args.dir)
    N, K = col["n"], col["ig_k"]
    mods = col["modules"][rank::world]
    log(f"gram: N={N} K={K}, {len(mods)} modules on this rank")
    G = torch.zeros(N, N, dtype=torch.float32, device=device)
    Bk = args.gram_block
    t0 = time.time()
    for path in mods:
        W = target.get_submodule(path).weight.detach()          # [d_out, d_in]
        W2t = (W * W).t().contiguous().bfloat16()               # [d_in, d_out]
        P = col[path]["p"].to(device)                           # [K, N, d_in] bf16
        Gm = col[path]["g"].to(device)                          # [K, N, d_out]
        for ka in range(K):
            for kb in range(K):
                pa, ga, pb, gb = P[ka], Gm[ka], P[kb], Gm[kb]
                for i0 in range(0, N, Bk):
                    pi, gi = pa[i0:i0 + Bk], ga[i0:i0 + Bk]
                    for j0 in range(0, N, Bk):
                        pj, gj = pb[j0:j0 + Bk], gb[j0:j0 + Bk]
                        pij = pi[:, None, :] * pj[None, :, :]           # [bi,bj,d_in]
                        t1 = pij @ W2t                                   # [bi,bj,d_out]
                        gij = gi[:, None, :] * gj[None, :, :]
                        G[i0:i0 + Bk, j0:j0 + Bk] += (t1 * gij).sum(-1).float() / (K * K)
        del P, Gm
        torch.cuda.empty_cache()
        log(f"gram {path} done ({time.time()-t0:.0f}s)")
    if ddp:
        dist.all_reduce(G)
    if rank == 0:
        torch.save(G.cpu(), args.dir / "gram.pt")
        d = G.diagonal()
        log(f"gram saved: diag mean {d.mean():.3e}, min {d.min():.3e} "
            f"({time.time()-t0:.0f}s)")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------------------------ factor ----

def normalize_gram(G: Tensor) -> Tensor:
    """Double-center (= center v's in feature space), then correlation-normalize."""
    rm = G.mean(0, keepdim=True)
    cm = G.mean(1, keepdim=True)
    Gc = G - rm - cm + G.mean()
    d = Gc.diagonal().clamp_min(1e-12).sqrt()
    return Gc / d[:, None] / d[None, :]


def spherical_kmeans(Y: Tensor, C: int, iters: int = 100, seed: int = 0,
                     restarts: int = 3):
    Y = F.normalize(Y, dim=1)
    best = None
    for r in range(restarts):
        gen = torch.Generator(device=Y.device).manual_seed(seed + r)
        # k-means++ style init on cosine distance (running max-sim, O(C·N·d))
        idx0 = torch.randint(Y.shape[0], (1,), generator=gen, device=Y.device)
        cents = [Y[idx0[0]]]
        simmax = Y @ cents[0]
        for _ in range(C - 1):
            probs = (1 - simmax).clamp_min(1e-6)
            c = Y[torch.multinomial(probs, 1, generator=gen)[0]]
            cents.append(c)
            simmax = torch.maximum(simmax, Y @ c)
        M = torch.stack(cents)
        for _ in range(iters):
            lab = (Y @ M.t()).argmax(1)
            Mn = torch.zeros_like(M).index_add_(0, lab, Y)
            cnt = torch.bincount(lab, minlength=C).clamp_min(1)
            M = F.normalize(Mn / cnt[:, None], dim=1)
        score = (Y @ M.t()).amax(1).sum().item()
        if best is None or score > best[0]:
            best = (score, lab.cpu(), M.cpu())
    return best[1], best[2]


def stage_factor(args):
    device = "cuda"
    kmax = max(1536, args.C + 256) if args.C > 0 else 1536
    eig_path = args.dir / "eig.pt"
    ck = None
    if eig_path.exists():
        ck = torch.load(eig_path, weights_only=True, map_location=device)
        if ck["evecs"].shape[1] < min(kmax, ck["N"] // 4):
            ck = None                                            # cached too few pairs
    if ck is not None:
        evals, evecs, N = ck["evals"], ck["evecs"], ck["N"]
        log(f"loaded cached eigenpairs (k={evecs.shape[1]})")
    else:
        G = torch.load(args.dir / "gram.pt", weights_only=True, map_location=device)
        N = G.shape[0]
        Gn = normalize_gram(G)
        del G
        Gn = 0.5 * (Gn + Gn.t())
        if N <= 8192:
            log("eigh...")
            evals, evecs = torch.linalg.eigh(Gn)
            evals, evecs = evals.flip(0), evecs.flip(1)          # descending
        else:
            # cusolver syevd fails at this size; we only need the top ~k pairs.
            # Randomized subspace iteration + Rayleigh-Ritz (symmetric input).
            k = min(kmax, N // 4)
            log(f"randomized top-{k} eigenpairs (subspace iteration)...")
            Q = torch.linalg.qr(torch.randn(N, k, device=Gn.device))[0]
            for it in range(8):
                Q = torch.linalg.qr(Gn @ Q)[0]
                log(f"  subspace iter {it}")
            T = Q.t() @ (Gn @ Q)
            ev, S = torch.linalg.eigh(0.5 * (T + T.t()))
            evals, evecs = ev.flip(0), (Q @ S).flip(1)
        del Gn
        torch.cuda.empty_cache()
        torch.save({"evals": evals.cpu(), "evecs": evecs.cpu(), "N": N}, eig_path)
    top = evals[:min(1500, N)].cpu()
    torch.save({"evals": evals[:min(4096, N)].cpu()}, args.dir / "spectrum.pt")
    lo, hi = 16, min(1024, N // 8, top.shape[0] - 1)
    gaps = (top[lo:hi] - top[lo + 1:hi + 1]) / top[lo:hi].clamp_min(1e-12)
    order = gaps.argsort(descending=True)[:8]
    log("eigengap candidates (C, relative gap): "
        + ", ".join(f"({lo + int(i)}, {gaps[i]:.3f})" for i in order))
    marks = [i for i in (256, 512, 1024) if i < top.shape[0]]
    log("spectrum head: " + ", ".join(f"{v:.3e}" for v in top[:12])
        + " ... " + " ".join(f"[{i}]={top[i]:.3e}" for i in marks))
    C = args.C if args.C > 0 else lo + int(order[0])
    log(f"factor: using C={C}")
    Y = evecs[:, :C].to(device) * evals[:C].to(device).clamp_min(0).sqrt()[None, :]
    labels, _ = spherical_kmeans(Y, C, seed=args.seed)
    sizes = torch.bincount(labels, minlength=C)
    log(f"cluster sizes: min {sizes.min()}, median {sizes.median()}, "
        f"max {sizes.max()}, empty {(sizes == 0).sum()}")
    torch.save({"labels": labels, "C": C, "sizes": sizes},
               args.dir / f"labels_C{C}.pt")
    if not (args.dir / "labels.pt").exists():
        torch.save({"labels": labels, "C": C, "sizes": sizes},
                   args.dir / "labels.pt")


# ----------------------------------------------------------------- extract ----

def stage_extract(args):
    device = "cuda"
    target = load_target(device)
    col = load_collect(args.dir)
    lab = torch.load(args.dir / "labels.pt", weights_only=True)
    labels, C = lab["labels"].to(device), int(lab["C"])
    N, K = col["n"], col["ig_k"]
    m = args.m
    banks, resid, report = {}, {}, {}
    for path in col["modules"]:
        W = target.get_submodule(path).weight.detach().float()   # [d_out, d_in]
        d_out, d_in = W.shape
        side = "g" if is_write_side(path) else "p"
        X = col[path][side].to(device).float().reshape(K * N, -1)  # [K*N, d_side]
        labs2 = labels.repeat(K)
        d_side = X.shape[1]
        anchors = torch.zeros(C, d_side, m, device=device)
        for c in range(C):
            Xc = X[labs2 == c]
            if Xc.shape[0] < m:                                   # degenerate cluster
                anchors[c] = torch.linalg.qr(torch.randn(d_side, m, device=device))[0]
                continue
            # top-m right singular vectors of the cluster's stacked vectors
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            anchors[c] = Vh[:m].t()
        A = anchors.permute(1, 0, 2).reshape(d_side, C * m)       # [d_side, C*m]
        S = A.t() @ A
        S += args.lam * S.diagonal().mean() * torch.eye(C * m, device=device)
        Sinv = torch.linalg.inv(S)
        if side == "p":       # W ~ sum_c B_c A_c^T, anchors on input side
            Bfree = W @ A @ Sinv                                  # [d_out, C*m]
            pieces_A = anchors                                    # [C, d_in, m]
            pieces_B = Bfree.reshape(d_out, C, m).permute(1, 2, 0)  # [C, m, d_out]
            recon = Bfree @ A.t()
        else:                  # W ~ sum_c U_c B_c, anchors on output side
            Bfree = Sinv @ A.t() @ W                              # [C*m, d_in]
            pieces_B = anchors.permute(0, 2, 1)                   # [C, m, d_out]
            pieces_A = Bfree.reshape(C, m, d_in).permute(0, 2, 1)  # [C, d_in, m]
            recon = A @ Bfree
        rho = W - recon
        banks[path] = {"A": pieces_A.cpu(), "B": pieces_B.cpu()}
        resid[path] = rho.cpu()
        report[path] = {"resid_frac": (rho.norm() / W.norm()).item() ** 2,
                        "side": side}
        log(f"extract {path}: resid mass {report[path]['resid_frac']:.3f}")
    masses = [r["resid_frac"] for r in report.values()]
    log(f"residual mass: mean {sum(masses)/len(masses):.3f}, "
        f"max {max(masses):.3f}")
    suf = f"_{args.banks_tag}" if args.banks_tag else ""
    torch.save({"banks": banks, "resid": resid, "C": C, "m": m,
                "modules": col["modules"], "report": report,
                "lam": args.lam}, args.dir / f"banks{suf}.pt")
    (args.dir / f"extract_report{suf}.json").write_text(json.dumps(report, indent=1))


# ------------------------------------------------------- partition extract ----

def stage_extract_p(args):
    """Partition extraction: every weight ENTRY assigned to exactly one cluster by
    per-capita attribution-mass argmax (|g|^T|p| matmuls); entries with flat
    cluster profile (max/mean < base_ratio) go to an always-on base component.
    Disjoint masses -> no cancellation between components by construction."""
    device = "cuda"
    target = load_target(device)
    col = load_extract_collect(args.dir)
    lab_path = (args.dir / f"labels_C{args.C}.pt" if args.C > 0
                and (args.dir / f"labels_C{args.C}.pt").exists()
                else args.dir / "labels.pt")
    lab = torch.load(lab_path, weights_only=True)
    labels, C = lab["labels"].to(device), int(lab["C"])
    log(f"extract_p: labels from {lab_path.name}, C={C}")
    N, K = col["n"], col["ig_k"]
    CH = 64                                    # cluster chunk (memory bound)
    assign, report = {}, {}
    for path in col["modules"]:
        W = target.get_submodule(path).weight.detach().float()
        d_out, d_in = W.shape
        module_col = load_extract_module(col, path)
        P = module_col["p"].to(device).float().abs().reshape(K * N, d_in)
        G = module_col["g"].to(device).float().abs().reshape(K * N, d_out)
        labs2 = labels.repeat(K)
        Wabs = W.abs()
        sumM = torch.zeros(d_out, d_in, device=device)
        mx = torch.zeros(d_out, d_in, device=device)
        am = torch.zeros(d_out, d_in, dtype=torch.int32, device=device)
        for c0 in range(0, C, CH):
            cc = min(CH, C - c0)
            M = torch.zeros(cc, d_out, d_in, device=device)
            for j in range(cc):
                sel = labs2 == (c0 + j)
                M[j] = (G[sel].t() @ P[sel]) / sel.sum().clamp_min(1)  # per-capita
            M *= Wabs[None]                                            # entry mass
            sumM += M.sum(0)
            m2, a2 = M.max(0)
            upd = m2 > mx
            am[upd] = (a2.to(torch.int32) + c0)[upd]
            mx[upd] = m2[upd]
            del M
        flat = mx / (sumM / C).clamp_min(1e-30) < args.base_ratio
        a = am + 1
        a[flat] = 0                                                    # 0 = base
        assign[path] = a.cpu()
        wmass = W.pow(2)
        base_frac = wmass[a == 0].sum().item() / wmass.sum().item()
        report[path] = {"base_frac": base_frac}
        log(f"extract_p {path}: base mass {base_frac:.3f}")
        del module_col, P, G, sumM, mx, am
        torch.cuda.empty_cache()
    suf = f"_{args.banks_tag}" if args.banks_tag else ""
    torch.save({"format": "partition", "assign": assign, "C": C, "m": 0,
                "modules": col["modules"], "report": report,
                "sensor": col["sensor"], "gim_tau": col["gim_tau"], "scalar": col.get("scalar", "ce"),
                "base_ratio": args.base_ratio}, args.dir / f"banks{suf}.pt")
    fr = [r["base_frac"] for r in report.values()]
    log(f"base mass: mean {sum(fr)/len(fr):.3f}")


# ---------------------------------------------------- fractional partition ----

def stage_extract_ps(args):
    """Fractional ownership: each entry split over its top-s clusters with
    weights proportional to per-capita attribution mass^(1/T). No base; masks
    sum to 1 per entry (exact faithfulness); no cancellation (positive slices)."""
    device = "cuda"
    target = load_target(device)
    col = load_extract_collect(args.dir)
    lab_path = (args.dir / f"labels_C{args.C}.pt" if args.C > 0
                and (args.dir / f"labels_C{args.C}.pt").exists()
                else args.dir / "labels.pt")
    lab = torch.load(lab_path, weights_only=True)
    labels, C = lab["labels"].to(device), int(lab["C"])
    log(f"extract_ps: labels from {lab_path.name}, C={C}, "
        f"T={args.soft_T}, s={args.soft_s}")
    N, K = col["n"], col["ig_k"]
    S, CH = args.soft_s, 32
    sidx, swgt, report = {}, {}, {}
    for path in col["modules"]:
        W = target.get_submodule(path).weight.detach().float()
        d_out, d_in = W.shape
        module_col = load_extract_module(col, path)
        P = module_col["p"].to(device).float().abs().reshape(K * N, d_in)
        G = module_col["g"].to(device).float().abs().reshape(K * N, d_out)
        labs2 = labels.repeat(K)
        vals = torch.zeros(S, d_out, d_in, device=device)
        idxs = torch.zeros(S, d_out, d_in, dtype=torch.int16, device=device)
        for c0 in range(0, C, CH):
            cc = min(CH, C - c0)
            M = torch.zeros(cc, d_out, d_in, device=device)
            for j in range(cc):
                sel = labs2 == (c0 + j)
                M[j] = (G[sel].t() @ P[sel]) / sel.sum().clamp_min(1)
            allv = torch.cat([vals, M])
            alli = torch.cat([idxs, (torch.arange(c0, c0 + cc, device=device,
                              dtype=torch.int16)[:, None, None]
                              .expand(-1, d_out, d_in))])
            top, sel_i = allv.topk(S, dim=0)
            vals = top
            idxs = alli.gather(0, sel_i)
            del M, allv, alli
        w = vals.clamp_min(0).pow(1.0 / args.soft_T)
        tot = w.sum(0, keepdim=True)
        w = w / tot.clamp_min(1e-30)
        w[0] = torch.where(tot[0] > 0, w[0], torch.ones_like(w[0]))  # dead entries -> top-1
        sidx[path] = idxs.cpu()
        swgt[path] = w.half().cpu()
        top1 = w[0].mean().item()
        report[path] = {"mean_top1_share": top1}
        log(f"extract_ps {path}: mean top-1 share {top1:.3f}")
        del module_col, P, G, vals, idxs, w
        torch.cuda.empty_cache()
    suf = f"_{args.banks_tag}" if args.banks_tag else ""
    torch.save({"format": "softpart", "sidx": sidx, "swgt": swgt, "C": C,
                "m": 0, "modules": col["modules"], "report": report,
                "sensor": col["sensor"], "gim_tau": col["gim_tau"], "scalar": col.get("scalar", "ce"),
                "soft_T": args.soft_T, "soft_s": S},
               args.dir / f"banks{suf}.pt")
    log(f"softpart banks saved ({suf or 'default'})")


# ---------------------------------------------------- gate-aware shares ----

def stage_extract_pg(args):
    """Gate-aware fractional shares: keep the top-s supports from a proportional
    softpart bank, but re-solve each entry's shares so that gated PARTIAL SUMS
    reconstruct the entry under the empirical gate distribution:
      min_s  E_t[ rho_te (sum_{c in S} g_tc s_c - 1)^2 ]   s.t.  sum_c s_c = 1
    with relevance rho_te = sum_k m_ek a_t,S_k (mass-weighted cluster attribution
    share). The equality constraint keeps exact faithfulness (all gates on = W).
    Only needs two moment tensors from a held-out stream:
      B[c,c']    = sum_t a_tc g_tc'
      Q[c,c',c''] = sum_t a_tc g_tc' g_tc''
    then per-entry 9x9 KKT solves, batched over entry chunks."""
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True     # moment sums, not model math
    target = load_target(device)
    src = torch.load(args.dir / f"banks_{args.src_banks_tag}.pt",
                     weights_only=True, map_location="cpu")
    assert src.get("format") == "softpart", "src must be a softpart bank"
    # gates/attribution come from src; supports + relevance masses come from
    # mass_banks_tag (default src) — for self-consistent iteration pass the
    # current gate-aware bank as src and the ORIGINAL proportional bank as mass
    # (solved shares are no longer attribution-mass estimates).
    mass = src
    if args.mass_banks_tag and args.mass_banks_tag != args.src_banks_tag:
        mass = torch.load(args.dir / f"banks_{args.mass_banks_tag}.pt",
                          weights_only=True, map_location="cpu")
        assert mass.get("format") == "softpart"
        assert mass["C"] == src["C"] and mass["soft_s"] == src["soft_s"]
        log(f"extract_pg: gates from {args.src_banks_tag}, "
            f"mass/supports from {args.mass_banks_tag}")
    run = GatedRunner(target, src, device)
    C, S = src["C"], src["soft_s"]
    from nano_param_decomp.pile_4L import make_loader
    loader = make_loader(args.batch_seqs, args.seq_len, 0, 1, "train",
                         args.seed + 555)            # distinct from eval stream
    Bmat = torch.zeros(C, C, device=device)
    Q = torch.zeros(C, C, C, device=device)
    r_mass = torch.zeros(C, device=device)
    t0 = time.time()
    CB = 64
    for b in range(args.gate_batches):
        idx = next(loader).to(device)
        attr, _ = run.attribution(idx, args.ig_k)
        attr = attr[:, 2:-2]
        a = (attr / attr.sum(-1, keepdim=True).clamp_min(1e-30)).flatten(0, 1)
        g = (attr / attr.amax(-1, keepdim=True).clamp_min(1e-30)
             > args.gate_thresh).float().flatten(0, 1)
        r_mass += a.sum(0)
        Bmat += a.t() @ g
        for c0 in range(0, C, CB):
            Q[c0:c0 + CB] += torch.einsum("tc,td,te->cde", a[:, c0:c0 + CB], g, g)
        log(f"gate stats batch {b}/{args.gate_batches} "
            f"({idx.shape[0] * (idx.shape[1] - 4)} toks, {time.time()-t0:.0f}s)")
    del run
    torch.cuda.empty_cache()

    EB = 65536
    eye = torch.eye(S, device=device)
    sidx_out, swgt_out, report = {}, {}, {}
    for path in src["modules"]:
        d_out, d_in = mass["sidx"][path].shape[1:]
        E = d_out * d_in
        Sx = mass["sidx"][path].reshape(S, E).t().long().to(device)  # [E, S]
        m = mass["swgt"][path].reshape(S, E).t().float().to(device)  # [E, S]
        s_new = torch.empty_like(m)
        obj0_sum = obj1_sum = rel_sum = 0.0
        n_fallback = 0
        for e0 in range(0, E, EB):
            Sc, mc = Sx[e0:e0 + EB], m[e0:e0 + EB]
            Qsel = Q[Sc[:, :, None, None], Sc[:, None, :, None], Sc[:, None, None, :]]
            A = (mc[:, :, None, None] * Qsel).sum(1)                 # [e, S, S]
            del Qsel
            Bsel = Bmat[Sc[:, :, None], Sc[:, None, :]]
            bv = (mc[:, :, None] * Bsel).sum(1)                      # [e, S]
            c0v = (mc * r_mass[Sc]).sum(-1)                          # [e]
            tr = A.diagonal(dim1=1, dim2=2).sum(-1)
            A = A + (1e-4 * tr[:, None, None] / S + 1e-12) * eye
            ne = A.shape[0]
            KKT = torch.zeros(ne, S + 1, S + 1, device=device)
            KKT[:, :S, :S] = A
            KKT[:, S, :S] = 1.0
            KKT[:, :S, S] = 1.0
            rhs = torch.cat([bv, torch.ones(ne, 1, device=device)], dim=1)
            sol = torch.linalg.solve(KKT, rhs)[:, :S].clamp_min(0)
            ssum = sol.sum(-1, keepdim=True)
            sol = torch.where(ssum > 1e-8, sol / ssum.clamp_min(1e-8), mc)
            dead = tr <= 1e-12                # support never relevantly active
            sol[dead] = mc[dead]
            n_fallback += dead.sum().item()

            def qobj(sv):
                return (torch.einsum("ei,eij,ej->e", sv, A, sv)
                        - 2 * (bv * sv).sum(-1) + c0v)
            obj0_sum += qobj(mc).sum().item()
            obj1_sum += qobj(sol).sum().item()
            rel_sum += c0v.sum().item()
            s_new[e0:e0 + EB] = sol
        sidx_out[path] = mass["sidx"][path]
        swgt_out[path] = s_new.t().reshape(S, d_out, d_in).half().cpu()
        report[path] = {
            "uncovered_frac_prop": obj0_sum / max(rel_sum, 1e-30),
            "uncovered_frac_solved": obj1_sum / max(rel_sum, 1e-30),
            "fallback_frac": n_fallback / E,
            "mean_top1_share": s_new.max(-1).values.mean().item()}
        log(f"extract_pg {path}: uncovered {report[path]['uncovered_frac_prop']:.3f}"
            f" -> {report[path]['uncovered_frac_solved']:.3f}, "
            f"fallback {report[path]['fallback_frac']:.3f}")
        del Sx, m, s_new
        torch.cuda.empty_cache()
    suf = f"_{args.banks_tag}" if args.banks_tag else ""
    torch.save({"format": "softpart", "sidx": sidx_out, "swgt": swgt_out, "C": C,
                "m": 0, "modules": src["modules"], "report": report,
                "sensor": src["sensor"], "gim_tau": src["gim_tau"], "scalar": src.get("scalar", "ce"),
                "soft_T": src["soft_T"], "soft_s": S,
                "share_mode": "gateaware", "src_banks": args.src_banks_tag,
                "mass_banks": args.mass_banks_tag or args.src_banks_tag,
                "gate_thresh": args.gate_thresh,
                "gate_batches": args.gate_batches},
               args.dir / f"banks{suf}.pt")
    u0 = [r["uncovered_frac_prop"] for r in report.values()]
    u1 = [r["uncovered_frac_solved"] for r in report.values()]
    log(f"gate-aware banks saved ({suf}): mean uncovered "
        f"{sum(u0)/len(u0):.3f} -> {sum(u1)/len(u1):.3f}")


# -------------------------------------------------------------------- eval ----

class GatedRunner:
    """Decomposed forward: out = rho x + sum_c gate_c * B_c(A_c^T x)  (+ exact by
    construction when all gates = 1). Also computes IG attribution inners."""

    CHUNK = 64          # cluster chunk for partition-mode einsums
    SOFT_CHUNK = 32     # cluster chunk for softpart mode (smaller at 1B scale)

    def __init__(self, target: nn.Module, bk: dict, device: str):
        self.target = target
        self.modules = bk["modules"]
        self.C, self.m = bk["C"], bk["m"]
        self.fmt = bk.get("format", "lowrank")
        self.scalar_mode = bk.get("scalar", "ce")
        if bk.get("sensor", "ig") == "gim":     # gates use the collect sensor
            apply_gim(target, bk.get("gim_tau", 2.0))
        if self.fmt == "partition":
            self.assign = {p: bk["assign"][p].to(device) for p in self.modules}
            # rho = always-on base slice of W; components = assign==c slices
            self.rho = {p: target.get_submodule(p).weight.detach()
                        * (self.assign[p] == 0) for p in self.modules}
        elif self.fmt == "softpart":
            # fractional ownership: entry e split over top-s clusters with
            # weights swgt (sum to 1 per entry); no base component
            self.sidx = {p: bk["sidx"][p].to(device) for p in self.modules}
            self.swgt = {p: bk["swgt"][p].to(device) for p in self.modules}
            # rho is identically zero in softpart; share one tensor per shape
            # (at 1B scale per-module zeros cost ~4GB of pure padding)
            zc: dict = {}
            for p in self.modules:
                w = target.get_submodule(p).weight
                if w.shape not in zc:
                    zc[w.shape] = torch.zeros_like(w)
            self.rho = {p: zc[target.get_submodule(p).weight.shape]
                        for p in self.modules}
        else:
            self.A = {p: bk["banks"][p]["A"].to(device) for p in self.modules}
            self.B = {p: bk["banks"][p]["B"].to(device) for p in self.modules}
            self.rho = {p: bk["resid"][p].to(device) for p in self.modules}
        self.mode = "target"
        self.wscale = 1.0
        self.gates: Tensor | None = None
        self.cache: dict[str, dict] = {}
        for path in self.modules:
            lin = target.get_submodule(path)
            lin._path = path
            if not hasattr(lin, "_orig_forward"):
                lin._orig_forward = lin.forward
            lin.forward = self._mk(lin)

    def _chunk_W(self, path: str, cids: Tensor, W: Tensor) -> Tensor:
        """[cc, d_out, d_in] per-component weight slices for clusters cids
        (0-indexed component ids)."""
        if self.fmt == "partition":
            return W[None] * (self.assign[path][None] == (cids[:, None, None] + 1))
        m = self.sidx[path][None] == cids[:, None, None, None].to(torch.int16)
        return W[None] * (m * self.swgt[path][None]).sum(1).to(W.dtype)

    def component_share(self, path: str, comps: Tensor) -> Tensor:
        """[d_out, d_in] fraction of each entry owned by the given components."""
        if self.fmt == "partition":
            return torch.isin(self.assign[path], comps.to(torch.int32) + 1).float()
        m = torch.isin(self.sidx[path], comps.to(torch.int16))
        return (m * self.swgt[path]).sum(0).float()

    def _mk(self, lin: nn.Linear):
        def fwd(x: Tensor) -> Tensor:
            path = lin._path
            if self.mode == "target":
                w = lin.weight if self.wscale == 1.0 else lin.weight * self.wscale
                out = F.linear(x, w, lin.bias)
                self.cache[path] = {"pre": x, "post": out}
                return out
            if self.fmt in ("partition", "softpart"):
                out = F.linear(x, self.rho[path], lin.bias)
                W = lin.weight
                ch = self.CHUNK if self.fmt == "partition" else self.SOFT_CHUNK
                for c0 in range(0, self.C, ch):
                    cids = torch.arange(c0, min(c0 + ch, self.C), device=W.device)
                    Wc = self._chunk_W(path, cids, W)                 # [cc,do,di]
                    g = self.gates[..., cids]                         # [B,T,cc]
                    out = out + torch.einsum("btd,cod,btc->bto", x, Wc, g)
                return out
            z = torch.einsum("btd,cdm->btcm", x, self.A[path])
            z = z * self.gates[..., None]
            out = torch.einsum("btcm,cmo->bto", z, self.B[path])
            out = out + F.linear(x, self.rho[path], lin.bias)
            return out
        return fwd

    def target_pass(self, idx: Tensor):
        self.mode, self.cache = "target", {}
        return self.target(idx), self.cache

    def gated_pass(self, idx: Tensor, gates: Tensor):
        self.mode, self.gates = "gated", gates
        logits = self.target(idx)
        self.mode = "target"
        return logits

    def attribution(self, idx: Tensor, ig_k: int) -> tuple[Tensor, Tensor]:
        """IG attribution A [B,T,C]: inner averaged over K scaled passes, then
        squared. Returns (attr, target_logits_at_full_scale)."""
        inner_tot, logits_full = None, None
        for k in range(1, ig_k + 1):
            self.wscale = k / ig_k
            logits, cache = self.target_pass(idx)
            s = scalar_sum(logits, idx, self.scalar_mode)
            posts = [cache[p]["post"] for p in self.modules]
            gposts = torch.autograd.grad(s, posts)
            inner = None
            for p, g in zip(self.modules, gposts, strict=True):
                pre = cache[p]["pre"].detach()
                if self.fmt in ("partition", "softpart"):
                    W = self.target.get_submodule(p).weight.detach()
                    ch = self.CHUNK if self.fmt == "partition" else self.SOFT_CHUNK
                    t = None
                    for c0 in range(0, self.C, ch):
                        cids = torch.arange(c0, min(c0 + ch, self.C),
                                            device=W.device)
                        Wc = self._chunk_W(p, cids, W)
                        tc = torch.einsum("btd,cod,bto->btc", pre, Wc, g)
                        t = tc if t is None else torch.cat([t, tc], dim=-1)
                else:
                    zi = torch.einsum("btd,cdm->btcm", pre, self.A[p])
                    gz = torch.einsum("bto,cmo->btcm", g, self.B[p])
                    t = (zi * gz).sum(-1)
                inner = t if inner is None else inner + t
            inner_tot = inner if inner_tot is None else inner_tot + inner
            if k == ig_k:
                logits_full = logits.detach()
        self.wscale = 1.0
        return (inner_tot / ig_k) ** 2, logits_full


def kl_per_token(lt: Tensor, lg: Tensor) -> float:
    return F.kl_div(F.log_softmax(lg[:, :-1].float(), -1),
                    F.log_softmax(lt[:, :-1].float(), -1),
                    log_target=True, reduction="batchmean").item() / lt.shape[1]


def kl_bt(lt: Tensor, lg: Tensor) -> float:
    """KL per token, averaged over batch*time."""
    p = F.log_softmax(lt[:, :-1].float(), -1)
    q = F.log_softmax(lg[:, :-1].float(), -1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def stage_eval(args):
    device = "cuda"
    suf = f"_{args.banks_tag}" if args.banks_tag else ""
    target = load_target(device)
    bk = torch.load(args.dir / f"banks{suf}.pt", weights_only=True,
                    map_location="cpu")
    run = GatedRunner(target, bk, device)
    from nano_param_decomp.pile_4L import make_loader
    loader = make_loader(args.batch_seqs, args.seq_len, 0, 1, "train",
                         args.seed + 999)                        # held-out stream
    C = bk["C"]
    if run.fmt in ("partition", "softpart"):
        faith = 0.0     # exact by construction: masks partition the entries of W
    else:
        faith = max((target.get_submodule(p).weight
                     - torch.einsum("cdm,cmo->od", run.A[p], run.B[p]) - run.rho[p]
                     ).abs().max().item() for p in bk["modules"])
    log(f"faithfulness identity max |err| = {faith:.2e} (should be ~fp32 eps)")

    KLs, KLres, KLoff, KLones, gpt, topj_kls = [], [], [], [], [], {}
    js = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for b in range(args.eval_batches):
        idx = next(loader).to(device)
        attr, lt = run.attribution(idx, args.ig_k)
        amax = attr.amax(-1, keepdim=True).clamp_min(1e-30)
        if args.gate_mode == "soft":
            gates = (attr / amax).pow(args.gate_tau)
        elif args.gate_mode == "ramp":
            # graded below tau*amax, saturates at fully-open above it;
            # binary gating is the gamma->inf limit
            gates = ((attr / amax) / args.gate_thresh).pow(args.gate_tau) \
                .clamp(max=1.0)
        elif args.gate_mode == "pr":
            # parameter-free: openness = attribution share scaled by the
            # token's effective component count (participation ratio), cap 1.
            # PR and share are scale-invariant; compute on max-normalized
            # attribution so a_c^2 sums cannot overflow fp32.
            an = attr / amax
            asum = an.sum(-1, keepdim=True).clamp_min(1e-30)
            pr = asum.pow(2) / an.pow(2).sum(-1, keepdim=True).clamp_min(1e-30)
            gates = (pr * an / asum).clamp(max=1.0)
        else:
            gates = (attr / amax > args.gate_thresh).float()
        gpt.append(gates.sum(-1).mean().item())
        with torch.no_grad():
            KLs.append(kl_bt(lt, run.gated_pass(idx, gates)))
            KLones.append(kl_bt(lt, run.gated_pass(idx, torch.ones_like(gates))))
            KLres.append(kl_bt(lt, run.gated_pass(idx, torch.zeros_like(gates))))
            # off baseline: rho zeroed too -> decomposed matrices contribute nothing
            if run.fmt == "softpart":
                KLoff.append(KLres[-1])   # rho already zero: off == residual-only
            else:
                rho_save = run.rho
                run.rho = {p: torch.zeros_like(r) for p, r in rho_save.items()}
                KLoff.append(kl_bt(lt, run.gated_pass(idx, torch.zeros_like(gates))))
                run.rho = rho_save
            if args.gate_mode == "hard":     # rank-based; identical across modes
                for j in js:
                    thr = attr.topk(min(j, C), dim=-1).values[..., -1:]
                    gj = (attr >= thr).float()
                    topj_kls.setdefault(j, []).append(
                        kl_bt(lt, run.gated_pass(idx, gj)))
        log(f"eval batch {b}: KL {KLs[-1]:.4f}, gates/tok {gpt[-1]:.1f}, "
            f"resid-only {KLres[-1]:.3f}, off {KLoff[-1]:.3f}, ones {KLones[-1]:.2e}")
    res = {
        "kl_full_gate": sum(KLs) / len(KLs),
        "kl_all_ones_sanity": sum(KLones) / len(KLones),
        "kl_residual_only": sum(KLres) / len(KLres),
        "kl_off_baseline": sum(KLoff) / len(KLoff),
        "gates_per_token": sum(gpt) / len(gpt),
        "keep_top_j": {j: sum(v) / len(v) for j, v in topj_kls.items()},
        "faith_max_abs_err": faith,
        "gate_mode": args.gate_mode, "gate_tau": args.gate_tau,
        "gate_thresh": args.gate_thresh, "C": C, "m": bk["m"],
        "resid_frac_mean": sum(r.get("resid_frac", r.get("base_frac", 0.0))
                               for r in bk["report"].values()) / len(bk["report"]),
    }
    ssuf = {"soft": f"_soft{args.gate_tau}", "ramp": f"_ramp{args.gate_tau}",
            "pr": "_pr"}.get(args.gate_mode, "")
    (args.dir / f"eval{suf}{ssuf}.json").write_text(json.dumps(res, indent=1))
    log("EVAL " + json.dumps(res, indent=1))


# ------------------------------------------------------------------ canary ----

def stage_canary(args):
    device = "cuda"
    suf = f"_{args.banks_tag}" if args.banks_tag else ""
    target = load_target(device)
    bk = torch.load(args.dir / f"banks{suf}.pt", weights_only=True,
                    map_location="cpu")
    run = GatedRunner(target, bk, device)
    C = bk["C"]
    gen = torch.Generator().manual_seed(args.seed)
    L = args.seq_len // 2
    idx = torch.randint(100, 50000, (args.batch_seqs, L), generator=gen)
    idx = torch.cat([idx, idx], dim=1).to(device)                # [B, 2L] repeated

    def copy_acc(logits):
        pred = logits[:, L:-1].argmax(-1)
        return (pred == idx[:, L + 1:]).float().mean().item()

    attr, lt = run.attribution(idx, args.ig_k)
    amax = attr.amax(-1, keepdim=True).clamp_min(1e-30)
    gates = (attr / amax > args.gate_thresh).float()
    with torch.no_grad():
        acc_t = copy_acc(lt)
        acc_g = copy_acc(run.gated_pass(idx, gates))
    # dgate contrast: second-half minus first-half mean attribution share
    share = attr / attr.sum(-1, keepdim=True).clamp_min(1e-30)
    contrast = (share[:, L:-1].mean((0, 1)) - share[:, 1:L].mean((0, 1)))
    order = contrast.argsort(descending=True)
    spec = (contrast[order[0]] / contrast[order[7]].clamp_min(1e-12)).item()
    res = {"target_copy_acc": acc_t, "gated_copy_acc": acc_g,
           "contrast_top8": order[:8].tolist(), "contrast_spec_1v8": spec,
           "ablate": {}}
    with torch.no_grad():
        for kk in [2, 8, 16, 64]:
            g2 = gates.clone(); g2[..., order[:kk]] = 0
            accs_r = []
            for s in range(3):
                rg = torch.Generator().manual_seed(s)
                rnd = torch.randperm(C, generator=rg)[:kk]
                g3 = gates.clone(); g3[..., rnd.to(device)] = 0
                accs_r.append(copy_acc(run.gated_pass(idx, g3)))
            res["ablate"][kk] = {
                "top": copy_acc(run.gated_pass(idx, g2)),
                "random_mean": sum(accs_r) / 3}
            log(f"canary ablate top-{kk}: {res['ablate'][kk]['top']:.3f} "
                f"vs random {res['ablate'][kk]['random_mean']:.3f}")
    (args.dir / f"canary{suf}.json").write_text(json.dumps(res, indent=1))
    log("CANARY " + json.dumps(res, indent=1))


# -------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "gram", "factor", "extract",
                                      "extract_p", "extract_ps", "extract_pg",
                                      "eval", "canary"])
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--n_positions", type=int, default=32768)
    ap.add_argument("--pos_per_seq", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_seqs", type=int, default=16)
    ap.add_argument("--ig_k", type=int, default=2)
    ap.add_argument("--gram_block", type=int, default=256)
    ap.add_argument("--C", type=int, default=0, help="0 = pick by eigengap")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--banks_tag", default="", help="suffix for banks/eval/canary "
                    "files (lambda-sweep arms share collect/gram/labels)")
    ap.add_argument("--sensor", choices=["ig", "gim"], default="ig",
                    help="gim = IG + GIM backward rules (softmax temperature "
                         "jacobian, frozen RMSNorm stats)")
    ap.add_argument("--gim_tau", type=float, default=2.0)
    ap.add_argument("--scalar", choices=["ce", "logp_pred", "logit_pred",
                                         "equal_reward"],
                    default="ce",
                    help="attribution scalar: ground-truth CE (default), "
                         "log-prob of the model's argmax token, or its raw "
                         "logit (saturation-free)")
    ap.add_argument("--base_ratio", type=float, default=3.0,
                    help="extract_p: entry goes to always-on base if its "
                         "max/mean cluster-mass ratio is below this")
    ap.add_argument("--gate_thresh", type=float, default=0.1)
    ap.add_argument("--gate_mode", choices=["hard", "soft", "ramp", "pr"],
                    default="hard")
    ap.add_argument("--gate_tau", type=float, default=1.0,
                    help="soft gates: g = (A/Amax)^tau; higher = sparser")
    ap.add_argument("--soft_T", type=float, default=1.0,
                    help="extract_ps: shares ~ mass^(1/T); T->0 = argmax")
    ap.add_argument("--soft_s", type=int, default=8,
                    help="extract_ps: top-s clusters per entry")
    ap.add_argument("--src_banks_tag", default="",
                    help="extract_pg: proportional softpart bank supplying "
                         "supports, mass weights, and bootstrap gates")
    ap.add_argument("--gate_batches", type=int, default=12,
                    help="extract_pg: batches of gate/attribution statistics")
    ap.add_argument("--mass_banks_tag", default="",
                    help="extract_pg: bank supplying supports + relevance "
                         "masses when iterating (default: src_banks_tag)")
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.dir = OUT_ROOT / args.tag
    torch.manual_seed(args.seed)
    {"collect": stage_collect, "gram": stage_gram, "factor": stage_factor,
     "extract": stage_extract, "extract_p": stage_extract_p,
     "extract_ps": stage_extract_ps, "extract_pg": stage_extract_pg,
     "eval": stage_eval, "canary": stage_canary}[args.stage](args)


if __name__ == "__main__":
    main()
