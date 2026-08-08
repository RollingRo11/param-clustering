"""Cluster WEIGHT COORDINATES by their effect profile, instead of positions.

The current fit clusters POSITIONS (rows of the feature matrix X, [N x D]) and
then derives per-weight ownership from which positions landed in which cluster.
Two weights end up together because the same inputs light them up — which is
co-attribution, and kmeans_ablation_study.py showed that reweighting the rows
does not change that.

The transpose clusters COLUMNS: coordinate j gets the profile
[a_1(j), ..., a_N(j)] of its attribution across inputs, and two coordinates join
the same component when deleting them would do similar things on the same
inputs. That is necessity by construction rather than by proxy.

It is also cheap, which was the open question. The column problem is 65,536
items in 8,160 dimensions — smaller than the row problem — so the k-means costs
seconds. The one real expense is downstream: a position-clustering gives each
component a HARD set of ~N/C positions, while a coordinate-clustering gives it
a dense weighting over all N, and the bank build is linear in how many
positions each component touches. Sparsifying that weighting to its top-m
positions puts the cost back in the same range.

Four arms, a 2x2 over what is clustered and whether magnitude survives:

    rows_sph     positions, L2-normalised     <- the current production method
    rows_euclid  positions, magnitude kept
    cols_sph     coordinates, L2-normalised
    cols_euclid  coordinates, magnitude kept  <- the necessity-shaped one

    python3.12 transpose_study.py --C 256 --positions 8192
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
from collect_fast_impl import pass_features, setup_model
from geo1m import load_spec
from german_vpd_1b import log, ranking_args
from kmeans_ablation_study import gini


def kmeans(Z, C, iters, seed, spherical):
    """k-means on rows of Z. spherical=True normalises rows and centroids."""
    g = torch.Generator(device=Z.device).manual_seed(seed)
    Y = F.normalize(Z, dim=1) if spherical else Z
    cent = Y[torch.randperm(Y.shape[0], generator=g, device=Y.device)[:C]].clone()
    for _ in range(iters):
        if spherical:
            lab = (Y @ cent.t()).argmax(1)
        else:
            lab = (Y.pow(2).sum(1, keepdim=True) - 2 * (Y @ cent.t())
                   + cent.pow(2).sum(1)[None]).argmin(1)
        sums = torch.zeros_like(cent).index_add_(0, lab, Y)
        cnt = torch.bincount(lab, minlength=C).float()[:, None]
        dead = cnt[:, 0] == 0
        if dead.any():
            sums[dead] = Y[torch.randperm(Y.shape[0], generator=g,
                                          device=Y.device)[:int(dead.sum())]]
            cnt[dead] = 1.0
        cent = F.normalize(sums, dim=1) if spherical else sums / cnt
    if spherical:
        lab = (Y @ cent.t()).argmax(1)
    else:
        lab = (Y.pow(2).sum(1, keepdim=True) - 2 * (Y @ cent.t())
               + cent.pow(2).sum(1)[None]).argmin(1)
    return cent, lab


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
    ap.add_argument("--top_positions", type=int, default=64,
                    help="sparsify each component's position weighting to its "
                         "top-m, so the bank build stays affordable")
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--eval_blocks", type=int, default=48)
    ap.add_argument("--sample_components", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="transpose_study.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    t00 = time.perf_counter()
    timing = {}

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    pile = torch.cat([data["pile_eval"], data["bio_retain_eval"]])
    seq = pile.shape[1]
    n_seq = min(pile.shape[0], max(1, args.positions // 64))
    per_seq = min(seq - 8, max(1, args.positions // n_seq))
    idx_fit = pile[:n_seq].to(dev)
    eval_idx = pile[:args.eval_blocks].to(dev)

    bank_meta = torch.load(run_dir / "banks_prop1b.pt", weights_only=True,
                           map_location="cpu", mmap=True)
    cfg = ranking_args(bank_meta)
    del bank_meta
    cap = setup_model(cfg, dev)
    spec, scales, dim = load_spec(run_dir, dev)
    mods = list(geo67.MODULES)

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
    X = torch.cat(PHI).to(dev).clamp(-6e4, 6e4)
    del PHI, cap
    torch.cuda.empty_cache()
    N = X.shape[0]
    timing["collect_features"] = round(time.perf_counter() - t00, 1)
    log(f"features [{N}, {dim}] in {timing['collect_features']}s")

    # ---- ROW arms: PCA embed then cluster positions (the current pipeline) ----
    t1 = time.perf_counter()
    Xc = X - X.mean(0)
    q = min(dim, N, args.embed_dim + 64)
    gg = torch.Generator(device=dev).manual_seed(args.seed)
    Q = torch.linalg.qr(torch.randn(dim, q, generator=gg, device=dev))[0]
    for _ in range(4):
        Q = torch.linalg.qr(Xc.t() @ (Xc @ Q))[0]
    Z = Xc @ Q
    small = (Z.t() @ Z) / N
    val, vec = torch.linalg.eigh(0.5 * (small + small.t()))
    order = val.argsort(descending=True)[:args.embed_dim]
    E = Xc @ (Q @ vec[:, order]).contiguous()
    del Xc, Z, small, val, vec, Q
    timing["pca_embed"] = round(time.perf_counter() - t1, 1)

    mu = {}          # per arm: [C, N] non-negative weighting over positions
    t1 = time.perf_counter()
    for name, sph in (("rows_sph", True), ("rows_euclid", False)):
        _, lab = kmeans(E, args.C, args.kmeans_iters, args.seed, sph)
        M = torch.zeros(args.C, N, device=dev)
        M[lab, torch.arange(N, device=dev)] = 1.0
        mu[name] = M / M.sum(1, keepdim=True).clamp_min(1e-30)
    timing["kmeans_rows"] = round(time.perf_counter() - t1, 1)

    # ---- COLUMN arms: cluster the D coordinates by their profile over N ----
    t1 = time.perf_counter()
    T = X.abs().t().contiguous()                       # [D, N]
    log(f"transposed problem: [{T.shape[0]}, {T.shape[1]}] "
        f"({T.numel() * 4 / 2**30:.2f} GiB)")
    for name, sph in (("cols_sph", True), ("cols_euclid", False)):
        cent, lab = kmeans(T, args.C, args.kmeans_iters, args.seed, sph)
        c = cent.clamp_min(0)
        # sparsify each component's position weighting to its top-m positions
        m = min(args.top_positions, N)
        v, i = c.topk(m, dim=1)
        M = torch.zeros(args.C, N, device=dev).scatter_(1, i, v)
        mu[name] = M / M.sum(1, keepdim=True).clamp_min(1e-30)
        sz = torch.bincount(lab, minlength=args.C)
        log(f"{name}: coordinate-cluster sizes min {int(sz.min())} "
            f"median {int(sz.median())} max {int(sz.max())}")
    timing["kmeans_cols"] = round(time.perf_counter() - t1, 1)
    del T, X, E
    torch.cuda.empty_cache()
    log(f"row k-means {timing['kmeans_rows']}s | "
        f"column k-means {timing['kmeans_cols']}s")

    # ---- banks: identical recipe, only mu differs ----
    target = geo1b.load_target_1b(dev)
    W0 = {p: target.get_submodule(p).weight.detach().clone() for p in mods}
    Wt = {p: target.get_submodule(p).weight for p in mods}
    banks = {}
    for name, M in mu.items():
        t1 = time.perf_counter()
        nz = [torch.nonzero(M[j], as_tuple=True)[0] for j in range(args.C)]
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
                Mm = torch.zeros(cc, d_out, d_in, device=dev)
                for j in range(cc):
                    r = nz[c0 + j]
                    if r.numel() == 0:
                        continue
                    w = M[c0 + j, r]
                    Mm[j] = (G[r] * w[:, None]).t() @ P[r]
                allv = torch.cat([vals, Mm])
                alli = torch.cat([idxs, torch.arange(
                    c0, c0 + cc, device=dev, dtype=torch.int16
                )[:, None, None].expand(-1, d_out, d_in)])
                top, si = allv.topk(S, dim=0)
                vals, idxs = top, alli.gather(0, si)
                del Mm, allv, alli
            w = vals.clamp_min(0)
            tot = w.sum(0, keepdim=True)
            w = w / tot.clamp_min(1e-30)
            w[0] = torch.where(tot[0] > 0, w[0], torch.ones_like(w[0]))
            sidx[p], swgt[p] = idxs, w.half()
            del P, G, vals, idxs, w
            torch.cuda.empty_cache()
        banks[name] = (sidx, swgt)
        timing[f"bank_{name}"] = round(time.perf_counter() - t1, 1)
        log(f"{name}: bank in {timing[f'bank_{name}']}s")
    del PG

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
            picks = torch.argsort(mass, descending=True)[
                torch.linspace(0, args.C - 1, args.sample_components).long()
            ].tolist()
            eff = []
            for c in picks:
                for p in mods:
                    mk = (swgt[p] * (sidx[p] == c)).sum(0, dtype=torch.float32)
                    Wt[p].copy_(W0[p] * (1.0 - mk))
                eff.append(ce() - base_ce)
            restore()
            order = torch.argsort(mass)
            rank = torch.empty(args.C, dtype=torch.int32, device=dev)
            rank[order] = torch.arange(args.C, dtype=torch.int32, device=dev)
            R = {p: rank[sidx[p].int()] for p in mods}
            curve = []
            for K in [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80,
                      96, 112, 128, 160, 192, 224, 256]:
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
            "gini_of_effects": round(gini(eff), 4),
            "max_single_effect": round(float(np.max(eff)), 5),
            "median_single_effect": round(float(np.median(eff)), 6),
            "total_effect": round(float(np.sum(eff)), 5),
        }
        r = results[name]
        log(f"{name:<12} removable {removable}/{args.C}  gini {r['gini_of_effects']:.3f}"
            f"  max {r['max_single_effect']:.4f}  median {r['median_single_effect']:.5f}")

    out = {"format": "transpose_study_v1", "C": args.C, "N": N, "D": dim,
           "base_ce": round(base_ce, 5), "timing_seconds": timing,
           "top_positions": args.top_positions, "results": results}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"timing: {json.dumps(timing)}")
    log(f"wrote {run_dir / args.out}  (total {time.perf_counter() - t00:.0f}s)")


if __name__ == "__main__":
    main()
