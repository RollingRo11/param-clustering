"""N=1M positions via exact-in-expectation random features (option 2).

The pairwise kernel is a sum over weight entries; importance-sampling D
coordinates (o,i) ~ W^2 makes the W^2 factor cancel exactly:
  k(x,x') = sum_oi W_oi^2 g_o g'_o p_i p'_i  ->  phi_r(x) = sqrt(Z/D) g_{o_r} p_{i_r}
  E[phi(x)^T phi(x')] = k(x,x')   (unbiased; IG handled by averaging the K passes)
Clustering becomes PCA + spherical k-means on [N, D] features (O(N*D), streamed);
extraction stays exact on a 1/32 stored (p,g) subset written in the standard
collect-shard format, so extract_ps/eval/german/emote run UNCHANGED on the tag.

Stages:
  spec     : sample the D feature coordinates ~ W^2 (two-stage row/col), save spec.pt
  validate : features from the stored N=16k collect vs the exact gram.pt; hard gate
  collect  : DDP; 64 positions/seq; feature memmap [N,D] fp16 + (p,g) subset shards
  cluster  : center, PCA to C dims, spherical k-means -> labels + subset labels file

  torchrun --nproc_per_node=2 geo1m.py collect --tag run1m ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/circuit-decomp/geo-attribution")

import numpy as np
import torch
import torch.nn.functional as F

import geo1b  # noqa: F401 — patches geo67 for the 1B target
import geo67
from geo67 import Capture, ddp_setup, log, normalize_gram, scalar_sum

SHM = Path("/dev/shm/geo1b")


# -------------------------------------------------------------------- spec ----

def stage_spec(args):
    """Importance-sample feature coordinates ~ W^2 * E[g^2] * E[p^2] (activation
    statistics from the stored N=16k collect) — near-optimal proposal for the
    heavy-tailed kernel sum; per-feature scales 1/sqrt(R_m * q) keep the
    estimator exactly unbiased."""
    device = "cuda"
    target = geo67.load_target(device)
    stats_src = SHM / "run1" / "collect_rank0.pt"
    col = torch.load(stats_src, weights_only=True, map_location="cpu")
    D = args.feat_dim
    Zq_m, rowW, mg2, mp2 = {}, {}, {}, {}
    for p in geo67.MODULES:
        W = target.get_submodule(p).weight.detach().float()
        Wsq = W * W
        g2 = col[p]["g"].float().pow(2).mean((0, 1)).to(device)   # [d_out]
        p2 = col[p]["p"].float().pow(2).mean((0, 1)).to(device)   # [d_in]
        mg2[p], mp2[p] = g2, p2
        rowW[p] = g2 * (Wsq @ p2)                                 # [d_out]
        Zq_m[p] = rowW[p].sum().item()
    Zq = sum(Zq_m.values())
    alloc = {p: int(D * Zq_m[p] / Zq) for p in geo67.MODULES}
    rem = D - sum(alloc.values())
    for p in sorted(geo67.MODULES,
                    key=lambda q: -(D * Zq_m[q] / Zq) % 1)[:rem]:
        alloc[p] += 1
    spec, scales = {}, {}
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for p in geo67.MODULES:
        R = alloc[p]
        if R == 0:
            spec[p] = (torch.zeros(0, dtype=torch.int32),
                       torch.zeros(0, dtype=torch.int32))
            scales[p] = torch.zeros(0)
            continue
        W = target.get_submodule(p).weight.detach().float()
        Wsq = W * W
        rows = torch.multinomial(rowW[p], R, replacement=True, generator=gen)
        colw = Wsq[rows] * mp2[p][None, :]                        # [R, d_in]
        cols = torch.multinomial(colw, 1, generator=gen).squeeze(1)
        # q(o,i | module) = rowW[o]/Zq_m * colw[o,i]/colw[o].sum()
        q = (rowW[p][rows] / Zq_m[p]) \
            * (colw[torch.arange(R, device=device), cols]
               / colw.sum(1).clamp_min(1e-30))
        # unbiased: contribution weight W_oi^2 / (R_m * q); feature carries
        # sqrt of it times |W| (so phi*phi' reproduces W^2 g g' p p')
        w_abs = W[rows, cols].abs()
        scales[p] = (w_abs / (R * q).clamp_min(1e-30).sqrt()).cpu()
        spec[p] = (rows.int().cpu(), cols.int().cpu())
    args.dir.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": spec, "scales": scales, "D": D, "alloc": alloc,
                "seed": args.seed, "Zq": Zq}, args.dir / "spec.pt")
    log(f"spec: D={D}, Zq={Zq:.4e}, stat-weighted proposal, "
        f"{sum(1 for r in alloc.values() if r)} modules")


def load_spec(dirp: Path, device):
    sp = torch.load(dirp / "spec.pt", weights_only=True, map_location="cpu")
    spec = {p: (o.long().to(device), i.long().to(device))
            for p, (o, i) in sp["spec"].items()}
    scales = {p: s.float().to(device) for p, s in sp["scales"].items()}
    return spec, scales, sp["D"]


def features_from(spec, scales, D, cache_g, cache_p, device):
    """cache_g/p: {path: [n, d]} single-pass tensors; returns [n, D] fp32."""
    outs = []
    for p, (o_idx, i_idx) in spec.items():
        if o_idx.numel() == 0:
            continue
        outs.append(cache_g[p][:, o_idx] * cache_p[p][:, i_idx]
                    * scales[p][None, :])
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------- validate ----

def stage_validate(args):
    device = "cuda"
    spec, scales, D = load_spec(args.dir, device)
    src = SHM / "run1" / "collect_rank0.pt"
    col = torch.load(src, weights_only=True, map_location="cpu")
    n, K = args.val_n, col["ig_k"]
    phi = None
    for k in range(K):
        g = {p: col[p]["g"][k, :n].to(device).float() for p in col["modules"]}
        pv = {p: col[p]["p"][k, :n].to(device).float() for p in col["modules"]}
        f = features_from(spec, scales, D, g, pv, device)
        phi = f if phi is None else phi + f
    phi /= K
    Ghat = phi @ phi.t()
    G = torch.load(SHM / "run1" / "gram.pt", weights_only=True,
                   map_location=device)[:n, :n]
    off = ~torch.eye(n, dtype=torch.bool, device=device)
    corr_raw = torch.corrcoef(torch.stack([G[off], Ghat[off]]))[0, 1].item()
    Gn, Ghn = normalize_gram(G), normalize_gram(Ghat)
    corr_n = torch.corrcoef(torch.stack([Gn[off], Ghn[off]]))[0, 1].item()
    rel = ((Ghat - G).norm() / G.norm()).item()
    res = {"corr_raw": corr_raw, "corr_normalized": corr_n,
           "rel_fro": rel, "D": D, "val_n": n}
    (args.dir / "validate.json").write_text(json.dumps(res, indent=1))
    log(f"VALIDATE corr_raw {corr_raw:.4f}, corr_normalized {corr_n:.4f}, "
        f"rel_fro {rel:.4f} (gate {args.val_gate})")
    if corr_n < args.val_gate:
        log("VALIDATION FAILED — aborting")
        sys.exit(1)


# ----------------------------------------------------------------- collect ----

def stage_collect(args):
    ddp, rank, world, device = ddp_setup()
    target = geo67.load_target(device)
    spec, scales, D = load_spec(args.dir, device)
    import nano_param_decomp.pile_4L as p4l
    loader = p4l.make_loader(args.batch_seqs * world, args.seq_len, rank, world,
                             "train", args.seed)
    cap = Capture(target)
    n_local = args.n_positions // world
    S, B, K = args.pos_per_seq, args.batch_seqs, args.ig_k
    ks = args.sub_per_seq
    n_batches = n_local // (B * S)
    gen = torch.Generator().manual_seed(args.seed + 7 * rank)
    feat = np.memmap(args.dir / f"feat_rank{rank}.f16", dtype=np.float16,
                     mode="w+", shape=(n_local, D))
    sub_store = {p: {"p": [[] for _ in range(K)], "g": [[] for _ in range(K)]}
                 for p in geo67.MODULES}
    sub_mask = torch.zeros(n_local, dtype=torch.bool)
    toks, nexts, sub_toks, sub_nexts = [], [], [], []
    t0 = time.time()
    for b in range(n_batches):
        idx = next(loader).to(device)
        Bc, T = idx.shape
        pos = torch.stack([torch.randperm(T - 6, generator=gen)[:S] + 4
                           for _ in range(Bc)]).to(device)
        bi = torch.arange(Bc, device=device)[:, None].expand(-1, S)
        phi_acc = torch.zeros(Bc * S, D, device=device)
        for k in range(1, K + 1):
            cap.wscale = k / K
            target.zero_grad(set_to_none=True)
            logits, cache = cap.run(idx)
            s = scalar_sum(logits, idx, args.scalar)
            posts = [cache[p]["post"] for p in geo67.MODULES]
            gposts = torch.autograd.grad(s, posts)
            gsel, psel = {}, {}
            for p, g in zip(geo67.MODULES, gposts, strict=True):
                pre = cache[p]["pre"].detach()
                psel[p] = pre[bi, pos].reshape(Bc * S, -1)
                gsel[p] = g[bi, pos].reshape(Bc * S, -1)
            phi_acc += features_from(spec, scales, D, gsel, psel, device) / K
            for p in geo67.MODULES:
                pk = psel[p].reshape(Bc, S, -1)[:, :ks]
                gk = gsel[p].reshape(Bc, S, -1)[:, :ks]
                sub_store[p]["p"][k - 1].append(
                    pk.reshape(Bc * ks, -1).bfloat16().cpu())
                sub_store[p]["g"][k - 1].append(
                    gk.reshape(Bc * ks, -1).bfloat16().cpu())
        lo = b * Bc * S
        feat[lo:lo + Bc * S] = phi_acc.clamp(-6e4, 6e4).cpu().numpy() \
            .astype(np.float16)
        m = torch.zeros(Bc, S, dtype=torch.bool)
        m[:, :ks] = True
        sub_mask[lo:lo + Bc * S] = m.reshape(-1)
        tk = idx[bi, pos].reshape(-1).cpu()
        nx = idx[bi, (pos + 1).clamp(max=T - 1)].reshape(-1).cpu()
        toks.append(tk)
        nexts.append(nx)
        sub_toks.append(tk.reshape(Bc, S)[:, :ks].reshape(-1))
        sub_nexts.append(nx.reshape(Bc, S)[:, :ks].reshape(-1))
        if b % 32 == 0:
            log(f"collect1m batch {b}/{n_batches} ({time.time()-t0:.0f}s)")
    feat.flush()
    out = {"modules": geo67.MODULES, "ig_k": K, "sensor": "ig",
           "gim_tau": 2.0, "scalar": args.scalar,
           "tok": torch.cat(sub_toks), "next": torch.cat(sub_nexts)}
    out["n"] = out["tok"].shape[0]
    for p in geo67.MODULES:
        out[p] = {"p": torch.stack([torch.cat(sub_store[p]["p"][k])
                                    for k in range(K)]),
                  "g": torch.stack([torch.cat(sub_store[p]["g"][k])
                                    for k in range(K)])}
    torch.save(out, args.dir / f"collect_rank{rank}.pt")
    torch.save({"sub_mask": sub_mask, "n_local": n_local,
                "tok": torch.cat(toks), "next": torch.cat(nexts)},
               args.dir / f"meta_rank{rank}.pt")
    log(f"collect1m done: {n_local} positions, {out['n']} in subset "
        f"({time.time()-t0:.0f}s)")
    if ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


# ----------------------------------------------------------------- cluster ----

def stage_cluster(args):
    device = "cuda"
    metas = [torch.load(args.dir / f"meta_rank{r}.pt", weights_only=True)
             for r in range(args.world)]
    n_locals = [m["n_local"] for m in metas]
    N = sum(n_locals)
    sp = torch.load(args.dir / "spec.pt", weights_only=True)
    D = sp["D"]
    X = torch.empty(N, D, dtype=torch.float16, device=device)
    off = 0
    for r, nl in enumerate(n_locals):
        mm = np.memmap(args.dir / f"feat_rank{r}.f16", dtype=np.float16,
                       mode="r", shape=(nl, D))
        for lo in range(0, nl, 131072):
            hi = min(lo + 131072, nl)
            X[off + lo:off + hi] = torch.from_numpy(
                np.ascontiguousarray(mm[lo:hi])).to(device)
        off += nl
    log(f"cluster: features loaded [{N}, {D}]")
    CH = 131072
    mean = torch.zeros(D, dtype=torch.float64, device=device)
    for lo in range(0, N, CH):
        mean += X[lo:lo + CH].double().sum(0)
    mean = (mean / N).float()
    for lo in range(0, N, CH):
        X[lo:lo + CH] = (X[lo:lo + CH].float() - mean).half()
    cov = torch.zeros(D, D, dtype=torch.float32, device=device)
    for lo in range(0, N, CH):
        xb = X[lo:lo + CH].float()
        cov += xb.t() @ xb
    cov /= N
    k = min(args.C + 128, D)
    log(f"cluster: randomized top-{k} eigenpairs of the {D}x{D} covariance")
    Q = torch.linalg.qr(torch.randn(D, k, device=device))[0]
    for _ in range(6):
        Q = torch.linalg.qr(cov @ Q)[0]
    Tm = Q.t() @ (cov @ Q)
    ev, Sv = torch.linalg.eigh(0.5 * (Tm + Tm.t()))
    V = (Q @ Sv).flip(1)[:, :args.C].contiguous()
    ev = ev.flip(0)[:args.C]
    log("spectrum head: " + ", ".join(f"{v:.3e}" for v in ev[:8])
        + f" ... [{args.C-1}]={ev[-1]:.3e}")
    Y = torch.empty(N, args.C, dtype=torch.float32, device=device)
    for lo in range(0, N, CH):
        Y[lo:lo + CH] = X[lo:lo + CH].float() @ V
    del X, cov
    torch.cuda.empty_cache()
    Y = F.normalize(Y, dim=1)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    subi = torch.randperm(N, generator=gen, device=device)[:65536]
    Ys = Y[subi]
    idx0 = torch.randint(Ys.shape[0], (1,), generator=gen, device=device)
    cents = [Ys[idx0[0]]]
    simmax = Ys @ cents[0]
    for _ in range(args.C - 1):
        probs = (1 - simmax).clamp_min(1e-6)
        cnew = Ys[torch.multinomial(probs, 1, generator=gen)[0]]
        cents.append(cnew)
        simmax = torch.maximum(simmax, Ys @ cnew)
    M = torch.stack(cents)
    lab = torch.empty(N, dtype=torch.int64, device=device)
    t0 = time.time()
    for it in range(args.kmeans_iters):
        for lo in range(0, N, CH):
            lab[lo:lo + CH] = (Y[lo:lo + CH] @ M.t()).argmax(1)
        Mn = torch.zeros_like(M)
        cnt = torch.zeros(args.C, device=device)
        for lo in range(0, N, CH):
            Mn.index_add_(0, lab[lo:lo + CH], Y[lo:lo + CH])
            cnt += torch.bincount(lab[lo:lo + CH], minlength=args.C).float()
        M = F.normalize(Mn / cnt.clamp_min(1)[:, None], dim=1)
        if it % 5 == 0:
            log(f"kmeans iter {it} ({time.time()-t0:.0f}s)")
    sizes = torch.bincount(lab, minlength=args.C)
    log(f"1M cluster sizes: min {sizes.min()}, median {sizes.median()}, "
        f"max {sizes.max()}, empty {(sizes == 0).sum()}")
    torch.save({"labels": lab.int().cpu(), "C": args.C,
                "sizes": sizes.cpu()}, args.dir / "labels_full.pt")
    masks = torch.cat([m["sub_mask"] for m in metas])
    labels_sub = lab.cpu()[masks].long()
    sub_sizes = torch.bincount(labels_sub, minlength=args.C)
    log(f"subset cluster sizes: min {sub_sizes.min()}, median "
        f"{sub_sizes.median()}, max {sub_sizes.max()}, "
        f"empty {(sub_sizes == 0).sum()}")
    torch.save({"labels": labels_sub, "C": args.C, "sizes": sub_sizes},
               args.dir / f"labels_C{args.C}.pt")
    log("cluster done: labels_full.pt + subset labels written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["spec", "validate", "collect", "cluster"])
    ap.add_argument("--tag", default="run1m")
    ap.add_argument("--feat_dim", type=int, default=16384)
    ap.add_argument("--val_n", type=int, default=2048)
    ap.add_argument("--val_gate", type=float, default=0.75)
    ap.add_argument("--n_positions", type=int, default=1048576)
    ap.add_argument("--pos_per_seq", type=int, default=64)
    ap.add_argument("--sub_per_seq", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_seqs", type=int, default=8)
    ap.add_argument("--ig_k", type=int, default=2)
    ap.add_argument("--scalar", default="ce")
    ap.add_argument("--C", type=int, default=2048)
    ap.add_argument("--kmeans_iters", type=int, default=30)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.dir = SHM / args.tag
    torch.manual_seed(args.seed)
    {"spec": stage_spec, "validate": stage_validate,
     "collect": stage_collect, "cluster": stage_cluster}[args.stage](args)


if __name__ == "__main__":
    main()
