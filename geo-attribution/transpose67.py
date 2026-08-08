"""The transpose study on VPD's own 67M target model.

The 4-layer Pile model from Goodfire's VPD paper (wandb goodfire/spd/t-9d2b8f02),
so the numbers are on the same target VPD reports on. Self-contained rather than
routed through geo67/collect_fast: that path is bound to the 1B Llama and the
67M is a different architecture (GPT2-style module tree, GELU MLP with a single
c_fc rather than SwiGLU, tied embeddings, plain rotate-half RoPE).

Everything that matters is matched to the production sensor:
  * GIM backward — tau=2 softmax Jacobian, dq/dk scaled by scaling/4, dv halved,
    RMSNorm rsqrt detached, c_fc pre-activation halved (the 1B halves gate and
    up; c_fc is the single analogue here)
  * equal_reward objective, PRE-softmax, one unit per ground-truth next token
  * features phi = g_o * p_i * |W_oi| / sqrt(R*q) at importance-sampled
    coordinates, the geo1m spec construction

24 decomposable matrices, ~28M parameters -- 34x smaller than the 1B.

    python3.12 transpose67.py --C 256 --positions 8192
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = Path("/dev/shm/geo67_target/model_step_99999.pt")
TOKENIZER = "EleutherAI/gpt-neox-20b"


def log(m):
    print(f"[geo67] {m}", flush=True)


# ------------------------------------------------------------------ model ----

class _Half(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        return g / 2.0


class _GimAttn(torch.autograd.Function):
    """Forward = ordinary causal SDPA. Backward = the GIM correction."""

    @staticmethod
    def forward(ctx, q, k, v, scaling, tau):
        ctx.save_for_backward(q, k, v)
        ctx.scaling, ctx.tau = scaling, tau
        s = (q @ k.transpose(-2, -1)) * scaling
        m = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).triu(1)
        s = s.masked_fill(m, float("-inf"))
        return torch.softmax(s.float(), -1).to(q.dtype) @ v

    @staticmethod
    def backward(ctx, go):
        q, k, v = ctx.saved_tensors
        s = ((q @ k.transpose(-2, -1)) * ctx.scaling).float()
        m = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).triu(1)
        s = s.masked_fill(m, float("-inf"))
        p = torch.softmax(s, -1).to(q.dtype)
        pt = torch.softmax(s / ctx.tau, -1).to(q.dtype)
        dp = go @ v.transpose(-2, -1)
        ds = (dp - (dp * pt).sum(-1, keepdim=True)) * pt
        dq = (ds @ k) * (ctx.scaling / 4.0)
        dk = (ds.transpose(-2, -1) @ q) * (ctx.scaling / 4.0)
        dv = (p.transpose(-2, -1) @ go) / 2.0
        return dq, dk, dv, None, None


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        # detached rsqrt: GIM freezes the normalisation statistics
        s = torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps).detach()
        return self.weight * (h * s).to(x.dtype)


def rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat((-x2, x1), -1) * sin


class Attn(nn.Module):
    def __init__(self, cfg, gim):
        super().__init__()
        d, h = cfg["n_embd"], cfg["n_head"]
        self.h, self.hd, self.gim = h, d // h, gim
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, nn.Linear(d, d, bias=False))

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        sc = 1.0 / math.sqrt(self.hd)
        if self.gim:
            o = _GimAttn.apply(q, k, v, sc, 2.0)
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg, gim):
        super().__init__()
        self.c_fc = nn.Linear(cfg["n_embd"], cfg["n_intermediate"], bias=False)
        self.down_proj = nn.Linear(cfg["n_intermediate"], cfg["n_embd"],
                                   bias=False)
        self.gim = gim

    def forward(self, x):
        h = self.c_fc(x)
        if self.gim:
            h = _Half.apply(h)
        return self.down_proj(F.gelu(h, approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg, gim):
        super().__init__()
        e = cfg["rms_norm_eps"]
        self.rms_1 = RMSNorm(cfg["n_embd"], e)
        self.attn = Attn(cfg, gim)
        self.rms_2 = RMSNorm(cfg["n_embd"], e)
        self.mlp = MLP(cfg, gim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.rms_1(x), cos, sin)
        return x + self.mlp(self.rms_2(x))


class Model67(nn.Module):
    def __init__(self, cfg, gim=True):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.h = nn.ModuleList([Block(cfg, gim) for _ in range(cfg["n_layer"])])
        self.ln_f = RMSNorm(cfg["n_embd"], cfg["rms_norm_eps"])
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        hd = cfg["n_embd"] // cfg["n_head"]
        inv = 1.0 / (cfg["rotary_base"] ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(cfg["n_ctx"]).float()
        f = torch.outer(t, inv)
        emb = torch.cat((f, f), -1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx)
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for b in self.h:
            x = b(x, cos, sin)
        return self.lm_head(self.ln_f(x))


MODULES = [f"h.{l}.{s}" for l in range(4) for s in
           ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
            "mlp.c_fc", "mlp.down_proj")]


def load67(device, gim=True):
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = {"vocab_size": 50277, "n_layer": 4, "n_head": 6, "n_embd": 768,
           "n_intermediate": 3072, "rotary_base": 10000.0,
           "rms_norm_eps": 1e-6, "n_ctx": 512}
    m = Model67(cfg, gim)
    sd = {k: v for k, v in sd.items()}
    sd["lm_head.weight"] = sd["wte.weight"]          # tied
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "cos" not in k and "sin" not in k], missing
    assert not unexpected, unexpected
    return m.float().to(device).eval()


# --------------------------------------------------------------- pipeline ----

def capture(model, idx, need_grad=True):
    """Return per-module (pre, grad-of-output) for the equal_reward objective."""
    pre, post = {}, {}
    hs = []
    for p in MODULES:
        mod = model.get_submodule(p)

        def hook(m, inp, out, _p=p):
            pre[_p] = inp[0]
            out.retain_grad()
            post[_p] = out
        hs.append(mod.register_forward_hook(hook))
    logits = model(idx)
    for h in hs:
        h.remove()
    reward = logits[:, :-1].float().gather(-1, idx[:, 1:, None]).sum()
    grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
    return ({p: pre[p].detach() for p in MODULES},
            {p: g.detach() for p, g in zip(MODULES, grads)}, logits.detach())


def build_spec(model, D, seed, device, P0, G0):
    """geo1m's importance-sampled spec, using pilot activation statistics."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rowW, Zq_m, mp2 = {}, {}, {}
    for p in MODULES:
        W = model.get_submodule(p).weight.detach().float()
        g2 = G0[p].float().pow(2).mean(0)
        p2 = P0[p].float().pow(2).mean(0)
        mp2[p] = p2
        rowW[p] = g2 * ((W * W) @ p2)
        Zq_m[p] = rowW[p].sum().item()
    Zq = sum(Zq_m.values())
    alloc = {p: max(1, int(D * Zq_m[p] / Zq)) for p in MODULES}
    spec, scales = {}, {}
    for p in MODULES:
        R = alloc[p]
        W = model.get_submodule(p).weight.detach().float()
        rows = torch.multinomial(rowW[p], R, replacement=True, generator=gen)
        colw = (W * W)[rows] * mp2[p][None, :]
        cols = torch.multinomial(colw, 1, generator=gen).squeeze(1)
        q = (rowW[p][rows] / Zq_m[p]) * (
            colw[torch.arange(R, device=device), cols]
            / colw.sum(1).clamp_min(1e-30))
        scales[p] = W[rows, cols].abs() / (R * q).clamp_min(1e-30).sqrt()
        spec[p] = (rows, cols)
    return spec, scales, sum(alloc.values())


def kmeans(Z, C, iters, seed, spherical):
    g = torch.Generator(device=Z.device).manual_seed(seed)
    Y = F.normalize(Z, dim=1) if spherical else Z
    cent = Y[torch.randperm(Y.shape[0], generator=g, device=Y.device)[:C]].clone()

    def assign(cn):
        if spherical:
            return (Y @ cn.t()).argmax(1)
        return (Y.pow(2).sum(1, keepdim=True) - 2 * (Y @ cn.t())
                + cn.pow(2).sum(1)[None]).argmin(1)
    for _ in range(iters):
        lab = assign(cent)
        sums = torch.zeros_like(cent).index_add_(0, lab, Y)
        cnt = torch.bincount(lab, minlength=C).float()[:, None]
        dead = cnt[:, 0] == 0
        if dead.any():
            sums[dead] = Y[torch.randperm(Y.shape[0], generator=g,
                                          device=Y.device)[:int(dead.sum())]]
            cnt[dead] = 1.0
        cent = F.normalize(sums, dim=1) if spherical else sums / cnt
    return cent, assign(cent)


def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = len(x)
    return 0.0 if x.sum() == 0 else float(
        (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--feat_dim", type=int, default=65536)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--top_positions", type=int, default=64)
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--sample_components", type=int, default=64,
                    help="components probed singly for the Gini; scale with C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("out/transpose67.json"))
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))

    # Ablation grid: the same fine spacing near zero as the C=256 study (the
    # region where the crossing lives), then log-spaced out to C.
    fine = [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128,
            160, 192, 224, 256]
    KGRID = sorted(set([k for k in fine if k <= args.C]
                       + [int(round(v)) for v in
                          np.geomspace(256, args.C, 64)] + [args.C]))
    KGRID = [k for k in KGRID if k <= args.C]

    t00 = time.perf_counter()
    timing = {}

    # ---- data: uncopyrighted Pile, the target's own tokenizer ----
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    need = (args.positions // 64 + args.eval_seqs + 8) * args.seq
    ds = load_dataset("monology/pile-uncopyrighted", split="train",
                      streaming=True)
    toks = []
    for r in ds:
        toks.extend(tok.encode(r["text"]))
        if len(toks) >= need + 1:
            break
    n_blk = len(toks) // args.seq
    IDS = torch.tensor(toks[:n_blk * args.seq]).view(n_blk, args.seq).to(dev)
    timing["data"] = round(time.perf_counter() - t00, 1)
    log(f"tokens: {IDS.shape} ({timing['data']}s)")

    model = load67(dev, gim=True)
    n_dec = sum(model.get_submodule(p).weight.numel() for p in MODULES)
    log(f"model loaded: {len(MODULES)} matrices, {n_dec / 1e6:.1f}M "
        f"decomposable params")

    fit_n = min(n_blk - args.eval_seqs, max(1, args.positions // 64))
    per_seq = min(args.seq - 8, max(1, args.positions // fit_n))
    fit_ids = IDS[:fit_n]
    eval_ids = IDS[-args.eval_seqs:]

    # ---- pilot pass: (p, g) at sampled positions, then the spec, then phi ----
    t1 = time.perf_counter()
    gcpu = torch.Generator().manual_seed(args.seed)
    avail = torch.arange(4, args.seq - 2)
    Pk, Gk = {p: [] for p in MODULES}, {p: [] for p in MODULES}
    for s in range(0, fit_n, 8):
        b = fit_ids[s:s + 8]
        if b.shape[0] == 0:
            continue
        sel = avail[torch.randperm(avail.numel(), generator=gcpu)[:per_seq]].to(dev)
        P, G, _ = capture(model, b)
        for p in MODULES:
            Pk[p].append(P[p][:, sel].reshape(-1, P[p].shape[-1]).half())
            Gk[p].append(G[p][:, sel].reshape(-1, G[p].shape[-1]).half())
        del P, G
    P0 = {p: torch.cat(v) for p, v in Pk.items()}
    G0 = {p: torch.cat(v) for p, v in Gk.items()}
    del Pk, Gk
    N = P0[MODULES[0]].shape[0]
    timing["capture"] = round(time.perf_counter() - t1, 1)
    log(f"captured {N} positions ({timing['capture']}s)")

    t1 = time.perf_counter()
    spec, scales, D = build_spec(model, args.feat_dim, args.seed, dev, P0, G0)
    X = torch.cat([(G0[p].float()[:, spec[p][0]] * P0[p].float()[:, spec[p][1]]
                    * scales[p][None]) for p in MODULES], 1).clamp(-6e4, 6e4)
    timing["features"] = round(time.perf_counter() - t1, 1)
    log(f"features [{N}, {X.shape[1]}] ({timing['features']}s)")

    # ---- four arms ----
    t1 = time.perf_counter()
    Xc = X - X.mean(0)
    q = min(X.shape[1], N, args.embed_dim + 64)
    gg = torch.Generator(device=dev).manual_seed(args.seed)
    Q = torch.linalg.qr(torch.randn(X.shape[1], q, generator=gg, device=dev))[0]
    for _ in range(4):
        Q = torch.linalg.qr(Xc.t() @ (Xc @ Q))[0]
    Z = Xc @ Q
    sm = (Z.t() @ Z) / N
    val, vec = torch.linalg.eigh(0.5 * (sm + sm.t()))
    E = Xc @ (Q @ vec[:, val.argsort(descending=True)[:args.embed_dim]]).contiguous()
    del Xc, Z, sm, val, vec, Q
    timing["pca"] = round(time.perf_counter() - t1, 1)

    mu = {}
    t1 = time.perf_counter()
    for name, sph in (("rows_sph", True), ("rows_euclid", False)):
        _, lab = kmeans(E, args.C, args.kmeans_iters, args.seed, sph)
        M = torch.zeros(args.C, N, device=dev)
        M[lab, torch.arange(N, device=dev)] = 1.0
        mu[name] = M / M.sum(1, keepdim=True).clamp_min(1e-30)
    timing["kmeans_rows"] = round(time.perf_counter() - t1, 1)
    t1 = time.perf_counter()
    T = X.abs().t().contiguous()
    for name, sph in (("cols_sph", True), ("cols_euclid", False)):
        cent, _ = kmeans(T, args.C, args.kmeans_iters, args.seed, sph)
        c = cent.clamp_min(0)
        m = min(args.top_positions, N)
        v, i = c.topk(m, dim=1)
        M = torch.zeros(args.C, N, device=dev).scatter_(1, i, v)
        mu[name] = M / M.sum(1, keepdim=True).clamp_min(1e-30)
    timing["kmeans_cols"] = round(time.perf_counter() - t1, 1)
    del T, X, E
    torch.cuda.empty_cache()
    log(f"k-means: rows {timing['kmeans_rows']}s, cols {timing['kmeans_cols']}s")

    # ---- banks + causal evaluation ----
    W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
    Wt = {p: model.get_submodule(p).weight for p in MODULES}

    @torch.no_grad()
    def ce():
        out = []
        for s in range(0, eval_ids.shape[0], 8):
            b = eval_ids[s:s + 8]
            lg = model(b)
            out.append(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1])
                                       .float(), b[:, 1:].reshape(-1)))
        return float(torch.stack(out).mean())

    @torch.no_grad()
    def restore():
        for p in MODULES:
            Wt[p].copy_(W0[p])

    restore()
    base_ce = ce()
    log(f"base CE {base_ce:.4f}")

    results = {}
    for name, M in mu.items():
        t1 = time.perf_counter()
        nz = [torch.nonzero(M[j], as_tuple=True)[0] for j in range(args.C)]
        sidx, swgt = {}, {}
        with torch.no_grad():
            for p in MODULES:
                Pa, Ga = P0[p].float().abs(), G0[p].float().abs()
                d_out, d_in = W0[p].shape
                S = args.soft_s
                vals = torch.zeros(S, d_out, d_in, device=dev)
                idxs = torch.zeros(S, d_out, d_in, dtype=torch.int16, device=dev)
                for c0 in range(0, args.C, 32):
                    cc = min(32, args.C - c0)
                    Mm = torch.zeros(cc, d_out, d_in, device=dev)
                    for j in range(cc):
                        r = nz[c0 + j]
                        if r.numel() == 0:
                            continue
                        Mm[j] = (Ga[r] * M[c0 + j, r][:, None]).t() @ Pa[r]
                    allv = torch.cat([vals, Mm])
                    alli = torch.cat([idxs, torch.arange(
                        c0, c0 + cc, device=dev, dtype=torch.int16
                    )[:, None, None].expand(-1, d_out, d_in)])
                    top, si = allv.topk(S, dim=0)
                    vals, idxs = top, alli.gather(0, si)
                w = vals.clamp_min(0)
                tot = w.sum(0, keepdim=True)
                w = w / tot.clamp_min(1e-30)
                w[0] = torch.where(tot[0] > 0, w[0], torch.ones_like(w[0]))
                sidx[p], swgt[p] = idxs, w.half()
                del vals, idxs, w
        timing[f"bank_{name}"] = round(time.perf_counter() - t1, 1)

        with torch.no_grad():
            mass = torch.zeros(args.C, device=dev, dtype=torch.float64)
            for p in MODULES:
                mass += torch.bincount(
                    sidx[p].reshape(-1).int(),
                    weights=(swgt[p].float() * (W0[p] ** 2)[None]
                             ).reshape(-1).double(), minlength=args.C)
            picks = torch.argsort(mass, descending=True)[
                torch.linspace(0, args.C - 1, args.sample_components).long()
            ].tolist()
            eff = []
            for c in picks:
                for p in MODULES:
                    mk = (swgt[p] * (sidx[p] == c)).sum(0, dtype=torch.float32)
                    Wt[p].copy_(W0[p] * (1.0 - mk))
                eff.append(ce() - base_ce)
            restore()
            order = torch.argsort(mass)
            rank = torch.empty(args.C, dtype=torch.int32, device=dev)
            rank[order] = torch.arange(args.C, dtype=torch.int32, device=dev)
            R = {p: rank[sidx[p].int()] for p in MODULES}
            curve = []
            for K in KGRID:
                for p in MODULES:
                    keep = (swgt[p] * (R[p] < K)).sum(0, dtype=torch.float32)
                    Wt[p].copy_(W0[p] * (1.0 - keep))
                curve.append({"k": K, "ce": round(ce(), 5)})
            restore()
        eff = np.array(eff)
        removable = max([r["k"] for r in curve
                         if r["ce"] - base_ce <= 0.05] or [0])
        results[name] = {"curve": curve, "removable_within_0.05": removable,
                         "gini_of_effects": round(gini(eff), 4),
                         "max_single_effect": round(float(eff.max()), 5),
                         "median_single_effect": round(float(np.median(eff)), 6),
                         "ablation_effects": [round(float(v), 6) for v in eff]}
        r = results[name]
        log(f"{name:<12} removable {removable}/{args.C}  gini {r['gini_of_effects']:.3f}"
            f"  max {r['max_single_effect']:.4f}  bank {timing[f'bank_{name}']}s")

    out = {"format": "transpose67_v1", "model": "VPD 4L-Pile 67M target "
           "(wandb goodfire/spd/t-9d2b8f02)", "C": args.C, "N": N,
           "D": args.feat_dim, "n_matrices": len(MODULES),
           "decomposable_params": n_dec, "base_ce": round(base_ce, 5),
           "timing_seconds": timing, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    timing["total"] = round(time.perf_counter() - t00, 1)
    log(f"timing {json.dumps(timing)}")
    log(f"wrote {args.out}  (total {timing['total']}s)")


if __name__ == "__main__":
    main()
