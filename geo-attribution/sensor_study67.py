"""Ablatability curves for different attribution sensors on VPD's 67M target.

The pipeline is held fixed -- production clustering (PCA -> spherical k-means on
token rows), production softpart bank, production minimality metric -- and ONLY
the attribution sensor is varied. The sensor is what turns a forward/backward
pass into the feature phi that gets clustered and into the ownership weights of
the bank, so it is applied end-to-end exactly as the production pipeline applies
GIM.

Sensors
-------
eap                 grad x act at the full model. Plain backward, one pass.
                    This is attribution patching / EAP / AtP.
ig2 ig3 ig5         Integrated gradients over K steps along the WEIGHT-SCALING
                    path: every decomposable matrix is scaled by a = k/K, the
                    pass is re-run, and g*p is averaged over the path. This is
                    the integration path this codebase already uses (--ig_k in
                    collect_fast_impl / geo67), i.e. a zero-weight baseline,
                    NOT an input-space or corrupted-activation baseline.
gim                 The production sensor (arXiv 2505.17630): four backward-only
                    modifications -- tau=2 softmax Jacobian, dq/dk scaled by
                    scaling/4 and dv halved, c_fc pre-activation gradient
                    halved, RMSNorm rsqrt detached. Mutually exclusive with IG.
gim_softmax_only    GIM ablation: the tau=2 softmax Jacobian alone.
gim_scales_only     GIM ablation: the other three modifications, true softmax
                    Jacobian. Complements gim_softmax_only.
actonly             |activation| only, gradient factor dropped. Control for
                    whether gradient information matters at all.
random              phi ~ N(0,1). The null floor: how well does an arbitrary
                    partition of the weights ablate?

Not the MIB benchmark. MIB's circuit-localization track scores methods against
ground-truth circuits on its own models and tasks (IOI, MCQA, arithmetic); the
67M Pile target is not one of them and has no such task. This measures the same
FAMILY of methods on our minimality metric instead.

    python3.12 sensor_study67.py --C 256 --seed 0
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

# mlp: "plain" = true GELU derivative; "half" = GIM's halved pre-activation
# gradient; "lin" = the equivalent linear map, i.e. the diagonal D with
# gelu(h) = D h exactly (D = gelu(h)/h), used in place of gelu'(h).
MODES = {
    "plain":   dict(tau=1.0, qk_div=1.0, v_div=1.0, mlp="plain", detach_rms=False),
    "gim":     dict(tau=2.0, qk_div=4.0, v_div=2.0, mlp="half",  detach_rms=True),
    "tau":     dict(tau=2.0, qk_div=1.0, v_div=1.0, mlp="plain", detach_rms=False),
    "scales":  dict(tau=1.0, qk_div=4.0, v_div=2.0, mlp="half",  detach_rms=True),
    "linmlp":  dict(tau=1.0, qk_div=1.0, v_div=1.0, mlp="lin",   detach_rms=False),
    "linmlp_gimattn": dict(tau=2.0, qk_div=4.0, v_div=2.0, mlp="lin",
                           detach_rms=False),
}
SENSORS = {
    "eap": dict(mode="plain", K=1),
    "ig2": dict(mode="plain", K=2),
    "ig3": dict(mode="plain", K=3),
    "ig5": dict(mode="plain", K=5),
    "gim": dict(mode="gim", K=1),
    "gim_softmax_only": dict(mode="tau", K=1),
    "gim_scales_only": dict(mode="scales", K=1),
    # MIB circuit-localization leaderboard, top entries
    "eap_linmlp_gimattn": dict(mode="linmlp_gimattn", K=1),   # rank 1
    "eap_linmlp": dict(mode="linmlp", K=1),                   # rank 4-ish
    "eapig_inputs5": dict(mode="plain", K=5, ig_path="inputs"),  # rank 3
    "actonly": dict(mode="plain", K=1, drop_grad=True),
    "random": dict(mode="plain", K=1, random=True),
}


def log(m):
    print(f"[sensor67] {m}", flush=True)


# ------------------------------------------------------------------ model ----

class _Scale(torch.autograd.Function):
    """Identity forward, gradient divided by `d` backward."""

    @staticmethod
    def forward(ctx, x, d):
        ctx.d = d
        return x

    @staticmethod
    def backward(ctx, g):
        return g / ctx.d, None


class _LinGelu(torch.autograd.Function):
    """GELU forward; backward uses the EQUIVALENT LINEAR MAP, not the derivative.

    For an elementwise sigma, the equivalent linear map at h is D = sigma(h)/h,
    the unique diagonal satisfying sigma(h) = D h exactly -- whereas sigma'(h)
    only matches to first order. For the tanh approximation,
        gelu(h) = 0.5 h (1 + tanh(c (h + 0.044715 h^3))),  c = sqrt(2/pi)
    so D = 0.5 (1 + tanh(...)) with no division and no h=0 singularity.
    """

    C = 0.7978845608028654  # sqrt(2/pi)

    @staticmethod
    def forward(ctx, h):
        d = 0.5 * (1.0 + torch.tanh(_LinGelu.C * (h + 0.044715 * h.pow(3))))
        ctx.save_for_backward(d)
        return h * d

    @staticmethod
    def backward(ctx, g):
        (d,) = ctx.saved_tensors
        return g * d


class _Attn(torch.autograd.Function):
    """Forward = ordinary causal SDPA. Backward = parameterised GIM correction.

    tau=1, qk_div=1, v_div=1 reproduces the exact softmax VJP, so the 'plain'
    mode is the true gradient and the only differences between modes are the
    intended ones.

    custom_fwd(cast_inputs=float32) pins this Function to fp32 under autocast:
    custom Functions otherwise see mixed dtypes in backward (saved bf16
    tensors vs fp32 incoming grads) and crash. Numerics unchanged -- this
    just opts the attention out of bf16.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, q, k, v, scaling, tau, qk_div, v_div):
        ctx.save_for_backward(q, k, v)
        ctx.scaling, ctx.tau = scaling, tau
        ctx.qk_div, ctx.v_div = qk_div, v_div
        s = (q @ k.transpose(-2, -1)) * scaling
        m = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).triu(1)
        s = s.masked_fill(m, float("-inf"))
        return torch.softmax(s.float(), -1).to(q.dtype) @ v

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, go):
        q, k, v = ctx.saved_tensors
        s = ((q @ k.transpose(-2, -1)) * ctx.scaling).float()
        m = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).triu(1)
        s = s.masked_fill(m, float("-inf"))
        p = torch.softmax(s, -1).to(q.dtype)
        pt = p if ctx.tau == 1.0 else torch.softmax(s / ctx.tau, -1).to(q.dtype)
        dp = go @ v.transpose(-2, -1)
        ds = (dp - (dp * pt).sum(-1, keepdim=True)) * pt
        dq = (ds @ k) * (ctx.scaling / ctx.qk_div)
        dk = (ds.transpose(-2, -1) @ q) * (ctx.scaling / ctx.qk_div)
        dv = (p.transpose(-2, -1) @ go) / ctx.v_div
        return dq, dk, dv, None, None, None, None


class RMSNorm(nn.Module):
    def __init__(self, d, eps, detach):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps, self.detach = eps, detach

    def forward(self, x):
        h = x.float()
        s = torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps)
        if self.detach:
            s = s.detach()
        return self.weight * (h * s).to(x.dtype)


def rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat((-x2, x1), -1) * sin


class Attn(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        d, h = cfg["n_embd"], cfg["n_head"]
        self.h, self.hd = h, d // h
        m = MODES[mode]
        self.tau, self.qk_div, self.v_div = m["tau"], m["qk_div"], m["v_div"]
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, n, nn.Linear(d, d, bias=False))

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.hd).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        sc = 1.0 / math.sqrt(self.hd)
        o = _Attn.apply(q, k, v, sc, self.tau, self.qk_div, self.v_div)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.c_fc = nn.Linear(cfg["n_embd"], cfg["n_intermediate"], bias=False)
        self.down_proj = nn.Linear(cfg["n_intermediate"], cfg["n_embd"],
                                   bias=False)
        self.mlp_mode = MODES[mode]["mlp"]

    def forward(self, x):
        h = self.c_fc(x)
        if self.mlp_mode == "lin":
            return self.down_proj(_LinGelu.apply(h))
        if self.mlp_mode == "half":
            h = _Scale.apply(h, 2.0)
        return self.down_proj(F.gelu(h, approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        e, dt = cfg["rms_norm_eps"], MODES[mode]["detach_rms"]
        self.rms_1 = RMSNorm(cfg["n_embd"], e, dt)
        self.attn = Attn(cfg, mode)
        self.rms_2 = RMSNorm(cfg["n_embd"], e, dt)
        self.mlp = MLP(cfg, mode)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.rms_1(x), cos, sin)
        return x + self.mlp(self.rms_2(x))


class Model67(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.h = nn.ModuleList([Block(cfg, mode) for _ in range(cfg["n_layer"])])
        self.ln_f = RMSNorm(cfg["n_embd"], cfg["rms_norm_eps"],
                            MODES[mode]["detach_rms"])
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        hd = cfg["n_embd"] // cfg["n_head"]
        inv = 1.0 / (cfg["rotary_base"] ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(cfg["n_ctx"]).float(), inv)
        emb = torch.cat((f, f), -1)
        self.register_buffer("cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None], persistent=False)

    def forward(self, idx, embed=None):
        T = idx.shape[1]
        x = self.wte(idx) if embed is None else embed
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for b in self.h:
            x = b(x, cos, sin)
        return self.lm_head(self.ln_f(x))


MODULES = [f"h.{l}.{s}" for l in range(4) for s in
           ("attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
            "mlp.c_fc", "mlp.down_proj")]


def load67(device, mode):
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = {"vocab_size": 50277, "n_layer": 4, "n_head": 6, "n_embd": 768,
           "n_intermediate": 3072, "rotary_base": 10000.0,
           "rms_norm_eps": 1e-6, "n_ctx": 512}
    m = Model67(cfg, mode)
    sd = dict(sd)
    sd["lm_head.weight"] = sd["wte.weight"]          # tied
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "cos" not in k and "sin" not in k], missing
    assert not unexpected, unexpected
    return m.float().to(device).eval()


# --------------------------------------------------------------- pipeline ----

def capture(model, idx, embed=None):
    pre, post, hs = {}, {}, []
    for p in MODULES:
        mod = model.get_submodule(p)

        def hook(m, inp, out, _p=p):
            pre[_p] = inp[0]
            out.retain_grad()
            post[_p] = out
        hs.append(mod.register_forward_hook(hook))
    logits = model(idx, embed=embed)
    for h in hs:
        h.remove()
    reward = logits[:, :-1].float().gather(-1, idx[:, 1:, None]).sum()
    grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
    return ({p: pre[p].detach() for p in MODULES},
            {p: g.detach() for p, g in zip(MODULES, grads)})


def build_spec(model, D, seed, device, p2m, g2m):
    gen = torch.Generator(device=device).manual_seed(seed)
    rowW, Zq_m = {}, {}
    for p in MODULES:
        W = model.get_submodule(p).weight.detach().float()
        rowW[p] = g2m[p] * ((W * W) @ p2m[p])
        Zq_m[p] = rowW[p].sum().item()
    Zq = sum(Zq_m.values())
    alloc = {p: max(1, int(D * Zq_m[p] / Zq)) for p in MODULES}
    spec, scales = {}, {}
    for p in MODULES:
        R = alloc[p]
        W = model.get_submodule(p).weight.detach().float()
        rows = torch.multinomial(rowW[p], R, replacement=True, generator=gen)
        colw = (W * W)[rows] * p2m[p][None, :]
        cols = torch.multinomial(colw, 1, generator=gen).squeeze(1)
        q = (rowW[p][rows] / Zq_m[p]) * (
            colw[torch.arange(R, device=device), cols]
            / colw.sum(1).clamp_min(1e-30))
        scales[p] = W[rows, cols].abs() / (R * q).clamp_min(1e-30).sqrt()
        spec[p] = (rows, cols)
    return spec, scales, sum(alloc.values())


def kmeans(Z, C, iters, seed, spherical=True):
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
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--sample_components", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensors", default=",".join(SENSORS))
    ap.add_argument("--out", type=Path, default=Path("out/sensor67.json"))
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))
    names = [s for s in args.sensors.split(",") if s]
    for n in names:
        assert n in SENSORS, f"unknown sensor {n}"
    t00 = time.perf_counter()
    timing = {}

    fine = [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128,
            160, 192, 224, 256]
    KGRID = sorted(set([k for k in fine if k <= args.C]
                       + [int(round(v)) for v in
                          np.geomspace(256, args.C, 64)] + [args.C]))
    KGRID = [k for k in KGRID if k <= args.C]

    # ---- data ----
    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    # stratified by meta.pile_set_name: equal block quota per subset.
    # Block shuffle fixed at seed 0, so --seed varies only the pilot
    # position sample and the k-means init, never the corpus.
    want_blocks = args.positions // 64 + args.eval_seqs + 8
    ids_cpu, blk_labels, pile_stats = load_pile_blocks(
        tok, want_blocks, args.seq, seed=0, tokenizer_name=TOKENIZER)
    IDS = ids_cpu.to(dev)
    n_blk = IDS.shape[0]
    timing["data"] = round(time.perf_counter() - t00, 1)
    log(f"tokens: {tuple(IDS.shape)} ({timing['data']}s)")

    fit_n = min(n_blk - args.eval_seqs, max(1, args.positions // 64))
    per_seq = min(args.seq - 8, max(1, args.positions // fit_n))
    fit_ids, eval_ids = IDS[:fit_n], IDS[-args.eval_seqs:]
    avail = torch.arange(4, args.seq - 2)

    results = {}
    base_ce = None
    for name in names:
        cfg = SENSORS[name]
        t_s = time.perf_counter()
        model = load67(dev, cfg["mode"])
        n_dec = sum(model.get_submodule(p).weight.numel() for p in MODULES)
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

        if base_ce is None:
            base_ce = ce()
            log(f"base CE {base_ce:.4f}   ({n_dec / 1e6:.1f}M decomposable)")

        # ---- pilot capture, once per IG path step ----
        K = cfg["K"]
        gcpu = torch.Generator().manual_seed(args.seed)
        sels = [avail[torch.randperm(avail.numel(), generator=gcpu)[:per_seq]]
                for _ in range(0, fit_n, 8)]
        Ps, Gs = [], []
        path = cfg.get("ig_path", "weights")
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for p in MODULES:
                    Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
            Pk = {p: [] for p in MODULES}
            Gk = {p: [] for p in MODULES}
            for bi, s in enumerate(range(0, fit_n, 8)):
                b = fit_ids[s:s + 8]
                if b.shape[0] == 0:
                    continue
                sel = sels[bi].to(dev)
                emb = None
                if path == "inputs":
                    # counterfactual = the neighbouring sequence in the batch
                    with torch.no_grad():
                        xc, xf = model.wte(b), model.wte(b.roll(1, 0))
                    emb = xf + a * (xc - xf)
                P, G = capture(model, b, embed=emb)
                for p in MODULES:
                    Pk[p].append(P[p][:, sel].reshape(-1, P[p].shape[-1]).half())
                    Gk[p].append(G[p][:, sel].reshape(-1, G[p].shape[-1]).half())
                del P, G
            Ps.append({p: torch.cat(v) for p, v in Pk.items()})
            Gs.append({p: torch.cat(v) for p, v in Gk.items()})
            del Pk, Gk
        restore()
        N = Ps[0][MODULES[0]].shape[0]

        gen = torch.Generator(device=dev).manual_seed(args.seed + 977)
        if cfg.get("drop_grad"):
            Gs = [{p: torch.ones_like(v) for p, v in g.items()} for g in Gs]
        if cfg.get("random"):
            Ps = [{p: torch.randn(v.shape, generator=gen, device=dev,
                                  dtype=v.dtype) for p, v in s.items()}
                  for s in Ps]
            Gs = [{p: torch.randn(v.shape, generator=gen, device=dev,
                                  dtype=v.dtype) for p, v in s.items()}
                  for s in Gs]
        timing[f"capture_{name}"] = round(time.perf_counter() - t_s, 1)

        # ---- spec from path-averaged second moments, then phi ----
        t1 = time.perf_counter()
        p2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Ps]).mean(0)
               for p in MODULES}
        g2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Gs]).mean(0)
               for p in MODULES}
        spec, scales, D = build_spec(model, args.feat_dim, args.seed, dev,
                                     p2m, g2m)
        X = torch.zeros(N, D, device=dev)
        off = 0
        for p in MODULES:
            r, c = spec[p]
            w = r.numel()
            acc = torch.zeros(N, w, device=dev)
            for st in range(K):
                acc += Gs[st][p].float()[:, r] * Ps[st][p].float()[:, c]
            X[:, off:off + w] = (acc / K) * scales[p][None]
            off += w
        X = X[:, :off].clamp(-6e4, 6e4)
        timing[f"features_{name}"] = round(time.perf_counter() - t1, 1)

        # ---- PCA -> spherical k-means on token rows (production arm) ----
        Xc = X - X.mean(0)
        q = min(X.shape[1], N, args.embed_dim + 64)
        gg = torch.Generator(device=dev).manual_seed(args.seed)
        Q = torch.linalg.qr(torch.randn(X.shape[1], q, generator=gg,
                                        device=dev))[0]
        for _ in range(4):
            Q = torch.linalg.qr(Xc.t() @ (Xc @ Q))[0]
        Z = Xc @ Q
        sm = (Z.t() @ Z) / N
        val, vec = torch.linalg.eigh(0.5 * (sm + sm.t()))
        E = Xc @ (Q @ vec[:, val.argsort(descending=True)[:args.embed_dim]]
                  ).contiguous()
        del Xc, Z, sm, val, vec, Q, X
        _, lab = kmeans(E, args.C, args.kmeans_iters, args.seed)
        M = torch.zeros(args.C, N, device=dev)
        M[lab, torch.arange(N, device=dev)] = 1.0
        M = M / M.sum(1, keepdim=True).clamp_min(1e-30)
        del E
        torch.cuda.empty_cache()

        # ---- softpart bank, ownership averaged over the same path ----
        t1 = time.perf_counter()
        nz = [torch.nonzero(M[j], as_tuple=True)[0] for j in range(args.C)]
        sidx, swgt = {}, {}
        with torch.no_grad():
            for p in MODULES:
                d_out, d_in = W0[p].shape
                S = args.soft_s
                vals = torch.zeros(S, d_out, d_in, device=dev)
                idxs = torch.zeros(S, d_out, d_in, dtype=torch.int16, device=dev)
                for c0 in range(0, args.C, 32):
                    cc = min(32, args.C - c0)
                    Mm = torch.zeros(cc, d_out, d_in, device=dev)
                    for st in range(K):
                        Pa = Ps[st][p].float().abs()
                        Ga = Gs[st][p].float().abs()
                        for j in range(cc):
                            r = nz[c0 + j]
                            if r.numel() == 0:
                                continue
                            Mm[j] += (Ga[r] * M[c0 + j, r][:, None]).t() @ Pa[r]
                    allv = torch.cat([vals, Mm / K])
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
        del Ps, Gs
        torch.cuda.empty_cache()

        # ---- minimality curve ----
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
            for Kk in KGRID:
                for p in MODULES:
                    keep = (swgt[p] * (R[p] < Kk)).sum(0, dtype=torch.float32)
                    Wt[p].copy_(W0[p] * (1.0 - keep))
                curve.append({"k": Kk, "ce": round(ce(), 5)})
            restore()
        eff = np.array(eff)
        removable = max([r["k"] for r in curve
                         if r["ce"] - base_ce <= 0.05] or [0])
        results[name] = {
            "mode": cfg["mode"], "ig_k": K, "curve": curve,
            "removable_within_0.05": removable,
            "gini_of_effects": round(gini(eff), 4),
            "max_single_effect": round(float(eff.max()), 5),
            "median_single_effect": round(float(np.median(eff)), 6),
            "ablation_effects": [round(float(v), 6) for v in eff]}
        timing[f"total_{name}"] = round(time.perf_counter() - t_s, 1)
        log(f"{name:<18} K={K}  removable {removable:>4}/{args.C}  "
            f"gini {results[name]['gini_of_effects']:.3f}  "
            f"max {results[name]['max_single_effect']:.4f}  "
            f"({timing[f'total_{name}']}s)")
        del model, W0, Wt, sidx, swgt, M, nz
        torch.cuda.empty_cache()

    out = {"format": "sensor67_v1", "model": "VPD 4L-Pile 67M target "
           "(wandb goodfire/spd/t-9d2b8f02)", "C": args.C, "N": N,
           "D": args.feat_dim, "seed": args.seed, "n_matrices": len(MODULES), "pile": pile_stats,
           "base_ce": round(base_ce, 5), "timing_seconds": timing,
           "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}  (total {time.perf_counter() - t00:.1f}s)")


if __name__ == "__main__":
    main()
