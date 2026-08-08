"""Does magnitude-weighted spherical k-means give more causal components?

The current fit L2-normalises every feature vector and runs spherical k-means,
so a position where the attribution is enormous and a position where it is
negligible pull on the centroids equally. Magnitude is exactly the part that
says "this matters", and causal_features.py showed magnitude tracks true
ablation effect at rho = 0.96. So the obvious change is to keep it as a SAMPLE
WEIGHT: same spherical geometry, same centroids-on-the-sphere, but each
position contributes in proportion to ||phi||.

This runs the whole pipeline twice, changing only that one line, and compares
the two decompositions on causal criteria rather than on cluster quality:

  ablation curve      how much can be deleted before CE moves. A
                      necessity-aligned partition should be able to shed more,
                      because it puts the causally-inert weights together.
  attribution->effect does per-component attribution predict that component's
                      measured ablation effect better under one fit?
  concentration       Gini of the ablation-effect distribution. Higher means
                      the partition separates "matters" from "does not".

Scale is reduced so the whole study runs in minutes: C=256, an 8k-position
pilot, the real GIM sensor and the real 112 matrices of Llama-3.2-1B.

    python3.12 kmeans_ablation_study.py --C 256 --positions 8192
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
import geo67
from collect_fast_impl import pass_features, setup_model, model_pass, objective
from geo1m import load_spec
from german_vpd_1b import log, ranking_args


def spherical_kmeans(Y, C, iters, seed, weights=None, log=print):
    """Spherical k-means on unit rows. `weights` scales each row's vote.

    weights=None reproduces the production fit exactly.
    """
    g = torch.Generator(device=Y.device).manual_seed(seed)
    perm = torch.randperm(Y.shape[0], generator=g, device=Y.device)
    cent = Y[perm[:C]].clone()
    w = None if weights is None else weights[:, None]
    for it in range(iters):
        lab = (Y @ cent.t()).argmax(1)
        src = Y if w is None else Y * w
        sums = torch.zeros_like(cent).index_add_(0, lab, src)
        empty = sums.norm(dim=1) == 0
        if empty.any():                       # re-seed dead centroids
            sums[empty] = Y[torch.randperm(Y.shape[0], generator=g,
                                           device=Y.device)[:int(empty.sum())]]
        cent = F.normalize(sums, dim=1)
    lab = (Y @ cent.t()).argmax(1)
    return cent, lab


def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--soft_T", type=float, default=1.0)
    ap.add_argument("--eval_blocks", type=int, default=16)
    ap.add_argument("--sample_components", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="kmeans_ablation_study.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    t00 = time.perf_counter()

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    pile = torch.cat([data["pile_eval"], data["bio_retain_eval"]])
    seq = pile.shape[1]
    # there are only ~48 eval blocks, so take more positions from each rather
    # than indexing past the end (which yields empty batches)
    n_seq = min(pile.shape[0], max(1, args.positions // 64))
    per_seq = min(seq - 8, max(1, args.positions // n_seq))
    idx_fit = pile[:n_seq].to(dev)
    log(f"pilot: {n_seq} sequences x {per_seq} positions = {n_seq * per_seq}")
    eval_idx = pile[:args.eval_blocks].to(dev)

    bank_meta = torch.load(run_dir / "banks_prop1b.pt", weights_only=True,
                           map_location="cpu", mmap=True)
    cfg = ranking_args(bank_meta)
    del bank_meta
    cap = setup_model(cfg, dev)
    spec, scales, dim = load_spec(run_dir, dev)
    mods = list(geo67.MODULES)

    # ---- collect features AND the (p, g) rows extract_ps needs ----
    gcpu = torch.Generator().manual_seed(args.seed)
    avail = torch.arange(4, seq - 2)
    PHI, PG = [], {p: {"p": [], "g": []} for p in mods}
    for s in range(0, n_seq, 4):
        b = idx_fit[s:s + 4]
        if b.shape[0] == 0:
            continue
        sel = avail[torch.randperm(avail.numel(), generator=gcpu)[:per_seq]]
        pos = sel[None].expand(b.shape[0], -1).to(dev)
        bi = torch.arange(b.shape[0], device=dev)[:, None].expand_as(pos)
        phi, sub = pass_features(cfg, cap, b, pos, bi, spec, scales, dim,
                                 keep_subset=per_seq)
        PHI.append(phi.float().cpu())
        for p in mods:
            PG[p]["p"].append(sub[p]["p"][0])
            PG[p]["g"].append(sub[p]["g"][0])
        del phi, sub
    X = torch.cat(PHI).to(dev)
    del PHI, cap
    torch.cuda.empty_cache()
    N = X.shape[0]
    log(f"features [{N}, {dim}] collected ({time.perf_counter() - t00:.0f}s)")

    # ---- PCA embed, exactly as fit_stream_model does ----
    X = X.clamp(-6e4, 6e4)
    mean = X.mean(0)
    Xc = X - mean
    q = min(dim, N, args.embed_dim + 64)
    gg = torch.Generator(device=dev).manual_seed(args.seed)
    Q = torch.linalg.qr(torch.randn(dim, q, generator=gg, device=dev))[0]
    for _ in range(4):
        Q = torch.linalg.qr(Xc.t() @ (Xc @ Q))[0]
    Z = Xc @ Q
    small = (Z.t() @ Z) / N
    val, vec = torch.linalg.eigh(0.5 * (small + small.t()))
    order = val.argsort(descending=True)[:args.embed_dim]
    projector = (Q @ vec[:, order]).contiguous()
    E = Xc @ projector
    norms = E.norm(dim=1)
    Y = F.normalize(E, dim=1)
    del Xc, Z, small, val, vec, Q, X
    log(f"embedded to {args.embed_dim}d; ||phi|| range "
        f"{float(norms.min()):.2f}..{float(norms.max()):.2f}, "
        f"ratio p99/p50 {float(norms.quantile(0.99) / norms.quantile(0.5)):.1f}")

    # `plain_seed2` is the control that decides the whole comparison: two runs
    # of the CURRENT method differing only in k-means init. Any gap between
    # plain and magweighted has to beat this to mean anything.
    fits = {
        "plain": spherical_kmeans(Y, args.C, args.kmeans_iters, args.seed),
        "plain_seed2": spherical_kmeans(Y, args.C, args.kmeans_iters,
                                        args.seed + 1000),
        "magweighted": spherical_kmeans(Y, args.C, args.kmeans_iters,
                                        args.seed, weights=norms),
        "magweighted_sq": spherical_kmeans(Y, args.C, args.kmeans_iters,
                                           args.seed, weights=norms ** 2),
    }
    for name, (_, lab) in fits.items():
        cnt = torch.bincount(lab, minlength=args.C).float()
        log(f"{name:<12} cluster sizes: min {int(cnt.min())} "
            f"median {int(cnt.median())} max {int(cnt.max())}, "
            f"{int((cnt == 0).sum())} empty")

    # ---- build a softpart bank per fit (same recipe as stage_extract_ps) ----
    target = geo1b.load_target_1b(dev)
    W0 = {p: target.get_submodule(p).weight.detach().clone() for p in mods}
    Wt = {p: target.get_submodule(p).weight for p in mods}
    banks = {}
    for name, (_, lab) in fits.items():
        t1 = time.perf_counter()
        sidx, swgt = {}, {}
        for p in mods:
            P = torch.cat(PG[p]["p"]).to(dev).float().abs()
            G = torch.cat(PG[p]["g"]).to(dev).float().abs()
            d_out, d_in = W0[p].shape
            S = args.soft_s
            vals = torch.zeros(S, d_out, d_in, device=dev)
            idxs = torch.zeros(S, d_out, d_in, dtype=torch.int16, device=dev)
            for c0 in range(0, args.C, 32):
                cc = min(32, args.C - c0)
                M = torch.zeros(cc, d_out, d_in, device=dev)
                for j in range(cc):
                    m = lab == (c0 + j)
                    M[j] = (G[m].t() @ P[m]) / m.sum().clamp_min(1)
                allv = torch.cat([vals, M])
                alli = torch.cat([idxs, torch.arange(
                    c0, c0 + cc, device=dev, dtype=torch.int16
                )[:, None, None].expand(-1, d_out, d_in)])
                top, si = allv.topk(S, dim=0)
                vals, idxs = top, alli.gather(0, si)
                del M, allv, alli
            w = vals.clamp_min(0).pow(1.0 / args.soft_T)
            tot = w.sum(0, keepdim=True)
            w = w / tot.clamp_min(1e-30)
            w[0] = torch.where(tot[0] > 0, w[0], torch.ones_like(w[0]))
            sidx[p], swgt[p] = idxs, w.half()
            del P, G, vals, idxs, w
            torch.cuda.empty_cache()
        banks[name] = (sidx, swgt)
        log(f"{name}: bank built ({time.perf_counter() - t1:.0f}s)")
    del PG

    # ---- causal evaluation ----
    @torch.no_grad()
    def ce():
        out = []
        for s in range(0, eval_idx.shape[0], 4):
            b = eval_idx[s:s + 4]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                lg = target(b)
            out.append(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1])
                                       .float(), b[:, 1:].reshape(-1)))
        return float(torch.stack(out).mean())

    @torch.no_grad()
    def restore():
        for p in mods:
            Wt[p].copy_(W0[p])

    restore()
    base_ce = ce()
    log(f"base CE {base_ce:.4f}")

    results = {}
    for name, (sidx, swgt) in banks.items():
        with torch.no_grad():
            mass = torch.zeros(args.C, device=dev, dtype=torch.float64)
            for p in mods:
                mass += torch.bincount(
                    sidx[p].reshape(-1).int(),
                    weights=(swgt[p].float() * (W0[p] ** 2)[None]
                             ).reshape(-1).double(), minlength=args.C)
            # single-component ablation effects
            picks = torch.argsort(mass, descending=True)[
                torch.linspace(0, args.C - 1, args.sample_components).long()
            ].tolist()
            eff = []
            for c in picks:
                for p in mods:
                    m = (swgt[p] * (sidx[p] == c)).sum(0, dtype=torch.float32)
                    Wt[p].copy_(W0[p] * (1.0 - m))
                eff.append(ce() - base_ce)
            restore()
            # minimality curve, lightest components first
            order = torch.argsort(mass)
            rank = torch.empty(args.C, dtype=torch.int32, device=dev)
            rank[order] = torch.arange(args.C, dtype=torch.int32, device=dev)
            R = {p: rank[sidx[p].int()] for p in mods}
            curve = []
            for K in [0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]:
                for p in mods:
                    keep = (swgt[p] * (R[p] < K)).sum(0, dtype=torch.float32)
                    Wt[p].copy_(W0[p] * (1.0 - keep))
                curve.append({"k": K, "ce": round(ce(), 5)})
            restore()
            del R
        eff = np.array(eff)
        removable = max([r["k"] for r in curve
                         if r["ce"] - base_ce <= 0.05] or [0])
        results[name] = {
            "curve": curve, "removable_within_0.05": removable,
            "ablation_effects": [round(float(v), 6) for v in eff],
            "sampled_components": picks,
            "gini_of_effects": round(gini(eff), 4),
            "max_single_effect": round(float(np.max(eff)), 5),
            "median_single_effect": round(float(np.median(eff)), 6),
        }
        r = results[name]
        log(f"{name:<12} removable within ΔCE .05: {removable}/{args.C}  | "
            f"gini {r['gini_of_effects']:.3f}  max single {r['max_single_effect']:.4f}"
            f"  median {r['median_single_effect']:.5f}")

    out = {"format": "kmeans_ablation_study_v1", "C": args.C, "N": N,
           "base_ce": round(base_ce, 5), "results": results,
           "norm_ratio_p99_p50": round(
               float(norms.quantile(0.99) / norms.quantile(0.5)), 3)}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}  (total {time.perf_counter() - t00:.0f}s)")


if __name__ == "__main__":
    main()
