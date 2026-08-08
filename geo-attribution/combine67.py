"""Can GIM and EAP be combined? Ensemble their per-input rankings.

The sufficiency metric is entirely about the ORDER in which components are
kept, so two sensors can be combined without touching the decomposition: build
the bank once, then for each target token score every component under each
sensor and merge the scores.

Merges (both scale-free, since GIM and EAP attributions differ in magnitude):
  zsum      z-score each sensor's attribution vector across components, add
  rankmean  average the two descending rank orders

Reported alongside each sensor alone and a random ranking of the same
components, so a merge that helps is visible against both parents.

    python3.12 combine67.py --base gim --sensors gim,eap --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sensor_study67 as S67
from sensor_study67 import MODULES, SENSORS, load67, capture, build_spec, kmeans


def log(m):
    print(f"[comb67] {m}", flush=True)


def build_bank(model, cfg, W0, Wt, fit_ids, fit_n, sels, args, dev):
    """Pilot -> features -> k-means -> softpart bank, for one sensor."""
    K, path = cfg["K"], cfg.get("ig_path", "weights")
    Ps, Gs = [], []
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
    with torch.no_grad():
        for p in MODULES:
            Wt[p].copy_(W0[p])
    N = Ps[0][MODULES[0]].shape[0]

    p2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Ps]).mean(0)
           for p in MODULES}
    g2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Gs]).mean(0)
           for p in MODULES}
    spec, scales, D = build_spec(model, args.feat_dim, args.seed, dev, p2m, g2m)
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
    Xc = X - X.mean(0)
    q = min(X.shape[1], N, args.embed_dim + 64)
    gg = torch.Generator(device=dev).manual_seed(args.seed)
    Q = torch.linalg.qr(torch.randn(X.shape[1], q, generator=gg, device=dev))[0]
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

    nz = [torch.nonzero(M[j], as_tuple=True)[0] for j in range(args.C)]
    sidx, swgt = {}, {}
    with torch.no_grad():
        for p in MODULES:
            d_out, d_in = W0[p].shape
            Sn = args.soft_s
            vals = torch.zeros(Sn, d_out, d_in, device=dev)
            idxs = torch.zeros(Sn, d_out, d_in, dtype=torch.int16, device=dev)
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
                top, si = allv.topk(Sn, dim=0)
                vals, idxs = top, alli.gather(0, si)
            w = vals.clamp_min(0)
            tot = w.sum(0, keepdim=True)
            w = w / tot.clamp_min(1e-30)
            w[0] = torch.where(tot[0] > 0, w[0], torch.ones_like(w[0]))
            sidx[p], swgt[p] = idxs, w.half()
    del Ps, Gs, M, nz
    torch.cuda.empty_cache()
    return sidx, swgt


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
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="gim",
                    help="sensor whose decomposition/bank is used")
    ap.add_argument("--sensors", default="gim,eap")
    ap.add_argument("--out", type=Path, default=Path("out/comb67.json"))
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))
    names = [s for s in args.sensors.split(",") if s]
    t00 = time.perf_counter()

    KEEP = [k for k in [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
                        256, 384, 512, 768, 1024] if k < args.C] + [args.C]

    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    # stratified by meta.pile_set_name: equal block quota per subset.
    # Block shuffle fixed at seed 0, so --seed varies only the pilot
    # position sample and the k-means init, never the corpus.
    want_blocks = args.positions // 64 + args.eval_seqs + 8
    ids_cpu, blk_labels, pile_stats = load_pile_blocks(
        tok, want_blocks, args.seq, seed=0, tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    n_blk = IDS.shape[0]
    fit_n = min(n_blk - args.eval_seqs, max(1, args.positions // 64))
    per_seq = min(args.seq - 8, max(1, args.positions // fit_n))
    fit_ids, eval_ids = IDS[:fit_n], IDS[-args.eval_seqs:]
    avail = torch.arange(4, args.seq - 2)
    gcpu = torch.Generator().manual_seed(args.seed)
    sels = [avail[torch.randperm(avail.numel(), generator=gcpu)[:per_seq]]
            for _ in range(0, fit_n, 8)]
    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_samples)]

    # one model per sensor; identical weights, different backward
    models = {n: load67(dev, SENSORS[n]["mode"]) for n in set(names + [args.base])}
    base_model = models[args.base]
    W0 = {p: base_model.get_submodule(p).weight.detach().clone()
          for p in MODULES}
    Wts = {n: {p: m.get_submodule(p).weight for p in MODULES}
           for n, m in models.items()}

    @torch.no_grad()
    def set_all(scale_from):
        for n in models:
            for p in MODULES:
                Wts[n][p].copy_(scale_from[p])

    @torch.no_grad()
    def tok_ce(b, t):
        lg = base_model(eval_ids[b:b + 1])[0, t].float()
        return float(F.cross_entropy(lg[None], eval_ids[b, t + 1][None]))

    set_all(W0)
    base_ce = float(np.mean([tok_ce(b, t) for b, t in samp]))
    log(f"unablated CE on {len(samp)} target tokens: {base_ce:.4f}")

    sidx, swgt = build_bank(base_model, SENSORS[args.base], W0,
                            Wts[args.base], fit_ids, fit_n, sels, args, dev)
    set_all(W0)
    log(f"bank built from '{args.base}'  ({time.perf_counter() - t00:.0f}s)")

    def attr_for(name, b, t):
        cfg = SENSORS[name]
        K, path = cfg["K"], cfg.get("ig_path", "weights")
        m, Wt = models[name], Wts[name]
        A = {p: torch.zeros_like(W0[p]) for p in MODULES}
        b_cf = (b + 1) % eval_ids.shape[0]
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for p in MODULES:
                    Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
            emb = None
            if path == "inputs":
                with torch.no_grad():
                    emb = m.wte(eval_ids[b_cf:b_cf + 1]) + a * (
                        m.wte(eval_ids[b:b + 1])
                        - m.wte(eval_ids[b_cf:b_cf + 1]))
            pre, post, hs = {}, {}, []
            for p in MODULES:
                mod = m.get_submodule(p)

                def hook(mm, inp, out, _p=p):
                    pre[_p] = inp[0]
                    out.retain_grad()
                    post[_p] = out
                hs.append(mod.register_forward_hook(hook))
            lg = m(eval_ids[b:b + 1], embed=emb)
            for h in hs:
                h.remove()
            reward = lg[0, t, eval_ids[b, t + 1]].float()
            grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
            for p, g in zip(MODULES, grads):
                A[p] += (g[0].float().t() @ pre[p][0].float()) / K
            del pre, post, grads, lg
        with torch.no_grad():
            for p in MODULES:
                Wt[p].copy_(W0[p])
        v = torch.zeros(args.C, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for p in MODULES:
                v += torch.bincount(
                    sidx[p].reshape(-1).int(),
                    weights=(swgt[p].float() * (A[p] * W0[p])[None]
                             ).reshape(-1).double(), minlength=args.C)
        del A
        return v

    def zs(v):
        return (v - v.mean()) / v.std().clamp_min(1e-30)

    def rk(v):                       # descending rank, 0 = most attributed
        o = torch.argsort(v, descending=True)
        r = torch.empty_like(v)
        r[o] = torch.arange(len(v), device=v.device, dtype=v.dtype)
        return r

    arms = list(names) + ["zsum", "rankmean", "random"]
    curves = {a: np.zeros((len(samp), len(KEEP))) for a in arms}
    rg = torch.Generator(device=dev).manual_seed(args.seed + 31)
    for si_, (b, t) in enumerate(samp):
        av = {n: attr_for(n, b, t) for n in names}
        orders = {n: torch.argsort(av[n], descending=True) for n in names}
        orders["zsum"] = torch.argsort(sum(zs(av[n]) for n in names),
                                       descending=True)
        orders["rankmean"] = torch.argsort(sum(rk(av[n]) for n in names))
        orders["random"] = torch.randperm(args.C, generator=rg, device=dev)
        for a in arms:
            rank = torch.empty(args.C, dtype=torch.int32, device=dev)
            rank[orders[a]] = torch.arange(args.C, dtype=torch.int32, device=dev)
            with torch.no_grad():
                R = {p: rank[sidx[p].int()] for p in MODULES}
                for ki, kk in enumerate(KEEP):
                    for p in MODULES:
                        keep = (swgt[p] * (R[p] < kk)).sum(0,
                                                          dtype=torch.float32)
                        Wts[args.base][p].copy_(W0[p] * keep)
                    curves[a][si_, ki] = tok_ce(b, t)
                del R
        set_all(W0)

    res = {}
    for a in arms:
        mu = curves[a].mean(0)
        res[a] = {"ce": [round(float(v), 5) for v in mu],
                  "ce_sem": [round(float(v), 5) for v in
                             curves[a].std(0, ddof=1) / np.sqrt(len(samp))],
                  "roundtrip_err_at_C": round(float(mu[-1] - base_ce), 6)}
        i16 = KEEP.index(16)
        log(f"{a:<12} CE@k=16 {mu[i16]:7.3f}   CE@k=32 "
            f"{mu[KEEP.index(32)]:7.3f}   roundtrip "
            f"{res[a]['roundtrip_err_at_C']:+.1e}")

    out = {"format": "comb67_v1", "C": args.C, "seed": args.seed,
           "base_decomposition": args.base, "sensors": names,
           "n_samples": len(samp), "base_token_ce": round(base_ce, 5),
           "pile": pile_stats,
           "keep_grid": KEEP, "results": res}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}  (total {time.perf_counter() - t00:.0f}s)")


if __name__ == "__main__":
    main()
