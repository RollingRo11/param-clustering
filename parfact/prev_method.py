"""The previous clustering method (geo-attribution), on the toy induction model.

Faithful miniature of combine67.build_bank / the 1B flagship pipeline:
  1. per-event attribution features over K=5 IG steps on the weights path
     (matrices scaled by alpha=(k+1)/K): signed mean-over-steps per-entry
     gradients, per-matrix RMS-normalized (the toy is small enough to skip
     the sketch/PCA stage — 2048 entries are used directly);
  2. hard k-means over prediction events (C clusters);
  3. per weight entry, cluster mass = mean over IG steps of the cluster-mean
     |gradient|; keep the top-8 clusters per entry, renormalize to sum 1
     (entries with zero mass assign their full share to the top cluster);
  4. components are elementwise shares of the weights: C_c = share_c * W,
     summing to W exactly.

Writes components.pt consumable by ablation_curve.py --components, so the
same per-token oracle curve runs on both decompositions:

    python prev_method.py --c_groups 100
    python ablation_curve.py --components out/prev_clustering_C100/components.pt \
        --orders per_example_asc:oracle
"""
import argparse
from pathlib import Path

import torch

from induction_model import InductionModel
from atoms import ATTN_MATRICES, collect_grads, make_events


def kmeans(x: torch.Tensor, c: int, iters: int, seed: int) -> torch.Tensor:
    """Plain euclidean k-means (random-row init, as in the prior pipeline).
    Returns hard labels [N]."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    cent = x[torch.randperm(x.shape[0], device=x.device, generator=g)[:c]]
    for _ in range(iters):
        d = torch.cdist(x, cent)
        lab = d.argmin(1)
        for j in range(c):
            m = lab == j
            if m.any():
                cent[j] = x[m].mean(0)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_groups", type=int, default=100)
    ap.add_argument("--ig_steps", type=int, default=5)
    ap.add_argument("--soft_s", type=int, default=8,
                    help="top clusters kept per weight entry")
    ap.add_argument("--n_seq", type=int, default=4096)
    ap.add_argument("--positions", default="mixed")
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    dev = args.device
    out_dir = args.out or (Path(__file__).parent
                           / f"out/prev_clustering_C{args.c_groups}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    matrices = list(ATTN_MATRICES)
    params = dict(model.named_parameters())
    w0 = {n: params[n].detach().clone() for n in matrices}

    # same event population the co-factorization was fit on
    events = make_events(model, args.n_seq, args.positions, seed=args.seed)
    seq, pos, y = events["seq"], events["pos"], events["y"]
    n_events = seq.shape[0]
    print(f"{n_events} events; IG K={args.ig_steps} on the weights path")

    # -- 1. IG-step gradients: signed mean for features, |.| mean for mass --
    feat = {n: torch.zeros(n_events, *w0[n].shape, device=dev)
            for n in matrices}
    mass = {n: torch.zeros(n_events, *w0[n].shape, device=dev)
            for n in matrices}
    with torch.no_grad():
        for step in range(args.ig_steps):
            alpha = (step + 1) / args.ig_steps
            for n in matrices:
                params[n].copy_(w0[n] * alpha)
            grads = collect_grads(model, matrices, seq, pos, y)
            for n in matrices:
                feat[n] += grads[n] / args.ig_steps
                mass[n] += grads[n].abs() / args.ig_steps
            del grads
            print(f"  IG step {step + 1}/{args.ig_steps} (alpha={alpha:.1f})",
                  flush=True)
        for n in matrices:
            params[n].copy_(w0[n])

    # -- 2. per-matrix RMS scaling -> k-means over events -------------------
    x = torch.cat([(feat[n] / feat[n].pow(2).mean().sqrt().clamp_min(1e-12))
                   .flatten(1) for n in matrices], dim=1)
    lab = kmeans(x, args.c_groups, args.kmeans_iters, args.seed)
    sizes = torch.bincount(lab, minlength=args.c_groups)
    print(f"k-means: {int((sizes > 0).sum())}/{args.c_groups} non-empty "
          f"clusters, largest {int(sizes.max())}")

    # -- 3. cluster attribution mass per entry -> top-8 shares --------------
    onehot = torch.zeros(args.c_groups, n_events, device=dev)
    onehot[lab, torch.arange(n_events, device=dev)] = 1.0
    onehot = onehot / onehot.sum(1, keepdim=True).clamp_min(1e-30)
    comps = {}
    for n in matrices:
        cm = onehot @ mass[n].flatten(1)              # [C, numel] cluster mean
        top, idx = cm.topk(args.soft_s, dim=0)        # [S, numel]
        share = top / top.sum(0, keepdim=True).clamp_min(1e-30)
        share[0] = torch.where(top.sum(0) > 0, share[0],
                               torch.ones_like(share[0]))
        dense = torch.zeros(args.c_groups, cm.shape[1], device=dev)
        dense.scatter_(0, idx, share)
        comps[n] = (dense * w0[n].flatten()[None]).reshape(-1, *w0[n].shape)
        err = (comps[n].sum(0) - w0[n]).abs().max().item()
        assert err < 1e-5, (n, err)

    torch.save({"components": {n: c.cpu() for n, c in comps.items()},
                "labels": lab.cpu(), "config": {k: str(v) for k, v
                                                in vars(args).items()}},
               out_dir / "components.pt")
    print(f"saved {out_dir}/components.pt "
          f"(sum_c C_c == W verified per matrix)")


if __name__ == "__main__":
    main()
