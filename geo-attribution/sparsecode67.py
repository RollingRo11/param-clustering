"""Sparse coding in place of k-means: same pipeline, sparse codes per token.

k-means constrains each token's code to be one-hot; here X ~ S.H with
||S_n||_0 <= T instead, fit by batched OMP + MOD dictionary updates in the same
256-dim PCA space the k-means ran in (shown insensitive to both sparsification
and higher PCA dims, so the space is not the bottleneck). The dictionary is
warm-started from the spherical k-means centroids: T=1 with hard assignment IS
the current method, so this is a strict relaxation.

The model is still queried exactly once (the attribution pass); OMP/MOD never
touch it -- the no-model-in-the-loop property is preserved.

Downstream, the softpart bank is built from |S|-weighted membership instead of
one-hot labels; shares still sum to 1 per weight entry, so k=C round-trip
stays exact. Two evaluation arms on the same 48 held-out tokens:

  rank    the standard protocol: rank ALL components by per-token attribution
          through the bank, keep top-k (apples-to-apples with the baseline)
  gate    the sparse coder's own active set: compute the eval token's phi,
          project, OMP against the dictionary, keep exactly its <=T support --
          the trained-gate-free analogue of VPD's ci>0

    python3.12 sparsecode67.py --C 4096 --T 16
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
    print(f"[sparse] {m}", flush=True)


def omp(E, H, T, exclude_dead=None):
    """Batched orthogonal matching pursuit. E [N,d], H [C,d] unit rows.

    Returns sel [N,T] atom ids and coef [N,T]."""
    N, d = E.shape
    dev = E.device
    sel = torch.zeros(N, T, dtype=torch.long, device=dev)
    coef = torch.zeros(N, T, device=dev)
    R = E.clone()
    I = torch.eye(T, device=dev)
    for t in range(T):
        corr = R @ H.t()
        if exclude_dead is not None:
            corr[:, exclude_dead] = 0
        if t > 0:
            corr.scatter_(1, sel[:, :t], 0.0)
        sel[:, t] = corr.abs().argmax(1)
        A = H[sel[:, :t + 1]]                                # [N,t+1,d]
        G = A @ A.transpose(1, 2) + 1e-5 * I[:t + 1, :t + 1]
        b = torch.einsum("nd,ntd->nt", E, A)
        c = torch.linalg.solve(G, b)
        R = E - torch.einsum("nt,ntd->nd", c, A)
        coef[:, :t + 1] = c
    return sel, coef


def fit_dict(E, C, T, iters, seed, warm=None):
    """OMP + MOD alternation. Returns H [C,d] unit rows, sel, coef."""
    dev = E.device
    g = torch.Generator(device=dev).manual_seed(seed)
    if warm is not None:
        H = F.normalize(warm.clone().float(), dim=1)
    else:
        H = F.normalize(E[torch.randperm(E.shape[0], generator=g,
                                         device=dev)[:C]].clone(), dim=1)
    N = E.shape[0]
    for it in range(iters):
        sel, coef = omp(E, H, T)
        S = torch.zeros(N, C, device=dev)
        S.scatter_(1, sel, coef)
        StS = S.t() @ S + 1e-4 * torch.eye(C, device=dev)
        H_new = torch.linalg.solve(StS, S.t() @ E)
        # dead atoms: reinit from the worst-reconstructed tokens
        used = (S != 0).any(0)
        err = (E - S @ H_new).pow(2).sum(1)
        bad = torch.argsort(err, descending=True)
        n_dead = int((~used).sum())
        if n_dead:
            H_new[~used] = E[bad[:n_dead]]
        H = F.normalize(H_new, dim=1)
        rel = float(err.sum() / E.pow(2).sum())
        log(f"  dict iter {it+1}/{iters}: resid {rel:.4f}, dead {n_dead}")
    sel, coef = omp(E, H, T)
    return H, sel, coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--C", type=int, default=4096)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--dict_iters", type=int, default=8)
    ap.add_argument("--sensor", default="ig5")
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--feat_dim", type=int, default=65536)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/sparsecode67.json"))
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))
    t00 = time.perf_counter()
    cfg = SENSORS[args.sensor]
    K, path = cfg["K"], cfg.get("ig_path", "weights")

    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    want = args.positions // 64 + args.eval_seqs + 8
    ids_cpu, _, _ = load_pile_blocks(tok, want, args.seq, seed=0,
                                     tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    fit_n = min(IDS.shape[0] - args.eval_seqs, max(1, args.positions // 64))
    per_seq = min(args.seq - 8, max(1, args.positions // fit_n))
    fit_ids, eval_ids = IDS[:fit_n], IDS[-args.eval_seqs:]
    avail = torch.arange(4, args.seq - 2)
    gcpu = torch.Generator().manual_seed(args.seed)
    sels = [avail[torch.randperm(avail.numel(), generator=gcpu)[:per_seq]]
            for _ in range(0, fit_n, 8)]

    model = load67(dev, cfg["mode"])
    W0 = {m: model.get_submodule(m).weight.detach().clone() for m in MODULES}
    Wt = {m: model.get_submodule(m).weight for m in MODULES}

    # ---- pilot (P,G) over the IG path, features, PCA (as sufficiency67) ----
    Ps, Gs = [], []
    for step in range(K):
        a = (step + 1) / K
        with torch.no_grad():
            for m in MODULES:
                Wt[m].copy_(W0[m] * (a if path == "weights" else 1.0))
        Pk = {m: [] for m in MODULES}
        Gk = {m: [] for m in MODULES}
        for bi, s in enumerate(range(0, fit_n, 8)):
            b = fit_ids[s:s + 8]
            if b.shape[0] == 0:
                continue
            sl = sels[bi].to(dev)
            P, G = capture(model, b)
            for m in MODULES:
                Pk[m].append(P[m][:, sl].reshape(-1, P[m].shape[-1]).half())
                Gk[m].append(G[m][:, sl].reshape(-1, G[m].shape[-1]).half())
            del P, G
        Ps.append({m: torch.cat(v) for m, v in Pk.items()})
        Gs.append({m: torch.cat(v) for m, v in Gk.items()})
    with torch.no_grad():
        for m in MODULES:
            Wt[m].copy_(W0[m])
    N = Ps[0][MODULES[0]].shape[0]
    p2m = {m: torch.stack([s[m].float().pow(2).mean(0) for s in Ps]).mean(0)
           for m in MODULES}
    g2m = {m: torch.stack([s[m].float().pow(2).mean(0) for s in Gs]).mean(0)
           for m in MODULES}
    spec, scales, D = build_spec(model, args.feat_dim, args.seed, dev,
                                 p2m, g2m)

    def phi_of(Pd, Gd):
        outs = []
        for m in MODULES:
            r, c = spec[m]
            acc = torch.zeros(Pd[0][m].shape[0], r.numel(), device=dev)
            for st in range(K):
                acc += Gd[st][m].float()[:, r] * Pd[st][m].float()[:, c]
            outs.append((acc / K) * scales[m][None])
        return torch.cat(outs, 1).clamp(-6e4, 6e4)

    X = phi_of(Ps, Gs)
    mu = X.mean(0)
    Xc = X - mu
    q = min(X.shape[1], N, args.embed_dim + 64)
    gg = torch.Generator(device=dev).manual_seed(args.seed)
    Q = torch.linalg.qr(torch.randn(X.shape[1], q, generator=gg, device=dev))[0]
    for _ in range(4):
        Q = torch.linalg.qr(Xc.t() @ (Xc @ Q))[0]
    Z = Xc @ Q
    sm = (Z.t() @ Z) / N
    val, vec = torch.linalg.eigh(0.5 * (sm + sm.t()))
    proj = (Q @ vec[:, val.argsort(descending=True)[:args.embed_dim]]
            ).contiguous()                                   # [D, edim]
    E = Xc @ proj
    del Xc, Z, sm, val, vec, Q, X
    torch.cuda.empty_cache()
    log(f"pilot N={N} D={D} -> E {tuple(E.shape)} "
        f"({time.perf_counter()-t00:.0f}s)")

    # ---- warm start from the current method, then sparse-code ----
    cent, _ = kmeans(E, args.C, args.kmeans_iters, args.seed)
    H, sel, coef = fit_dict(E, args.C, args.T, args.dict_iters, args.seed,
                            warm=cent)
    used_per_tok = float((coef != 0).sum(1).float().mean())
    log(f"dictionary fit: mean atoms/token {used_per_tok:.1f} of T={args.T}")

    # ---- bank from |S|-weighted membership ----
    t1 = time.perf_counter()
    M = torch.zeros(args.C, N, device=dev)
    M.scatter_(0, sel.t(), coef.abs().t())
    M = M / M.sum(1, keepdim=True).clamp_min(1e-30)
    nz = [torch.nonzero(M[j], as_tuple=True)[0] for j in range(args.C)]
    sidx, swgt = {}, {}
    with torch.no_grad():
        for m in MODULES:
            d_out, d_in = W0[m].shape
            Sn = args.soft_s
            vals = torch.zeros(Sn, d_out, d_in, device=dev)
            idxs = torch.zeros(Sn, d_out, d_in, dtype=torch.int16, device=dev)
            for c0 in range(0, args.C, 32):
                cc = min(32, args.C - c0)
                Mm = torch.zeros(cc, d_out, d_in, device=dev)
                for st in range(K):
                    Pa = Ps[st][m].float().abs()
                    Ga = Gs[st][m].float().abs()
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
            sidx[m], swgt[m] = idxs, w.half()
    del Ps, Gs, M
    torch.cuda.empty_cache()
    log(f"bank built ({time.perf_counter()-t1:.0f}s)")

    # ---- eval ----
    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_samples)]
    KEEP = [k for k in [0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768,
                        1024, 1536, 2048, 3072] if k < args.C] + [args.C]

    @torch.no_grad()
    def tok_ce(b, t):
        lg = model(eval_ids[b:b + 1])[0, t].float()
        return float(F.cross_entropy(lg[None], eval_ids[b, t + 1][None]))

    base = float(np.mean([tok_ce(b, t) for b, t in samp]))
    log(f"unablated CE on {len(samp)} eval tokens: {base:.4f}")

    def token_capture(b, t):
        """(P,G) at one eval position over the IG path, plus A*W."""
        Pd = [{} for _ in range(K)]
        Gd = [{} for _ in range(K)]
        A = {m: torch.zeros_like(W0[m]) for m in MODULES}
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for m in MODULES:
                    Wt[m].copy_(W0[m] * (a if path == "weights" else 1.0))
            pre, post, hs = {}, {}, []
            for m in MODULES:
                mod = model.get_submodule(m)

                def hook(mm, inp, out, _m=m):
                    pre[_m] = inp[0]
                    out.retain_grad()
                    post[_m] = out
                hs.append(mod.register_forward_hook(hook))
            lg = model(eval_ids[b:b + 1])
            for h in hs:
                h.remove()
            reward = lg[0, t, eval_ids[b, t + 1]].float()
            grads = torch.autograd.grad(reward, [post[m] for m in MODULES])
            for m, g in zip(MODULES, grads):
                Pd[step][m] = pre[m][0, t][None].half()
                Gd[step][m] = g[0, t][None].half()
                A[m] += (g[0].float().t() @ pre[m][0].float()) / K
            del pre, post, grads, lg
        with torch.no_grad():
            for m in MODULES:
                Wt[m].copy_(W0[m])
        return Pd, Gd, {m: A[m] * W0[m] for m in MODULES}

    curves = np.zeros((len(samp), len(KEEP)))
    gate_ce = np.zeros(len(samp))
    gate_k = np.zeros(len(samp))
    for j, (b, t) in enumerate(samp):
        Pd, Gd, AW = token_capture(b, t)
        # arm A: rank all components by attribution through the bank
        attr = torch.zeros(args.C, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for m in MODULES:
                attr += torch.bincount(
                    sidx[m].reshape(-1).int(),
                    weights=(swgt[m].float() * AW[m][None]
                             ).reshape(-1).double(), minlength=args.C)
        order = torch.argsort(attr, descending=True)
        rank = torch.empty(args.C, dtype=torch.int32, device=dev)
        rank[order] = torch.arange(args.C, dtype=torch.int32, device=dev)
        with torch.no_grad():
            R = {m: rank[sidx[m].int()] for m in MODULES}
            for ki, kk in enumerate(KEEP):
                for m in MODULES:
                    keep = (swgt[m].float() * (R[m] < kk)
                            ).sum(0, dtype=torch.float32)
                    Wt[m].copy_(W0[m] * keep)
                curves[j, ki] = tok_ce(b, t)
            del R
            for m in MODULES:
                Wt[m].copy_(W0[m])
        # arm B: the coder's own gate -- OMP support of this token's phi
        e = (phi_of([{m: Pd[st][m] for m in MODULES} for st in range(K)],
                    [{m: Gd[st][m] for m in MODULES} for st in range(K)])
             - mu[None]) @ proj
        s_sel, s_coef = omp(e, H, args.T)
        support = s_sel[0][s_coef[0] != 0]
        gate_k[j] = support.numel()
        actb = torch.zeros(args.C, dtype=torch.bool, device=dev)
        actb[support] = True
        with torch.no_grad():
            for m in MODULES:
                keep = (swgt[m].float() * actb[sidx[m].int()]
                        ).sum(0, dtype=torch.float32)
                Wt[m].copy_(W0[m] * keep)
            gate_ce[j] = tok_ce(b, t)
            for m in MODULES:
                Wt[m].copy_(W0[m])
        del Pd, Gd, AW

    mu_c = curves.mean(0)
    thr = next((k for k, v in zip(KEEP, mu_c) if v - base <= 0.25), args.C)
    out = {"format": "sparsecode67_v1", "C": args.C, "T": args.T,
           "sensor": args.sensor, "dict_iters": args.dict_iters,
           "n_samples": len(samp), "base_token_ce": round(base, 5),
           "keep_grid": KEEP,
           "rank": {"ce": [round(float(v), 5) for v in mu_c],
                    "k_within_0.25": thr,
                    "roundtrip_err_at_C": round(float(mu_c[-1] - base), 6)},
           "gate": {"mean_ce": round(float(gate_ce.mean()), 5),
                    "median_ce": round(float(np.median(gate_ce)), 5),
                    "mean_support": round(float(gate_k.mean()), 2)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log(f"rank arm: k needed {thr}/{args.C}  CE@64 "
        f"{mu_c[KEEP.index(64)]:.2f}  CE@256 {mu_c[KEEP.index(256)]:.2f}  "
        f"roundtrip {out['rank']['roundtrip_err_at_C']:+.1e}")
    log(f"gate arm: keep only the token's own {gate_k.mean():.1f} atoms -> "
        f"CE {gate_ce.mean():.2f} (median {np.median(gate_ce):.2f}, "
        f"unablated {base:.2f})")
    log(f"wrote {args.out} (total {time.perf_counter()-t00:.0f}s)")


if __name__ == "__main__":
    main()
