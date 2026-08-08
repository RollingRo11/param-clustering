"""Activation statistics for the frozen decomposition.

Two different questions get conflated when people ask "how active is a
component":

  * cluster mass  - what fraction of token positions land in a component when
    each position is assigned to its nearest centroid (hard argmax). Sums to 1
    across components; mean is exactly 1/C.
  * posterior concentration - for one token position, how much of the softmax
    posterior sits on the top component, and how many components carry
    appreciable mass. This one depends on the rank temperature, which is an
    analysis knob rather than a property of the decomposition, so it is
    reported across several temperatures.

Also reports the static side: what fraction of the model's squared weight mass
each component owns under the soft partition.

  python3.12 decomp_stats.py --batches 120 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from collect_fast_impl import (make_loader, pass_features, sampled_batch,
                               setup_model)
from geo1m import load_spec
from streaming_decomposition import load_stream_model
from german_vpd_1b import log, ranking_args

TEMPS = (0.02, 0.05, 0.10)
THRESH = (0.01, 0.02, 0.05, 0.10)
RANKS = (1, 2, 4, 8, 16, 32, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--batches", type=int, default=120)
    parser.add_argument("--batch_seqs", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pos_per_seq", type=int, default=506)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--data_path", type=Path, default=geo1b.BIN_PATH)
    parser.add_argument("--synthetic_data", action="store_true")
    parser.add_argument("--data_order", default="sequential")
    parser.add_argument("--out", default="decomp_stats.json")
    args = parser.parse_args()
    device = args.device
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":")[1]))
    run_dir = args.artifact_root / args.tag
    torch.manual_seed(args.seed)

    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    meta = {k: bank[k] for k in ("format", "C", "sensor", "gim_tau", "scalar")
            if k in bank}
    # Static side: squared-weight mass each component owns, from the soft
    # partition (shares sum to 1 per weight entry).
    target_cpu = None
    own = torch.zeros(int(bank["C"]), dtype=torch.float64)
    total_mass = 0.0
    cfg = ranking_args(meta)
    cap = setup_model(cfg, device)
    for path in bank["modules"]:
        w = cap.target.get_submodule(path).weight.detach().float().cpu() ** 2
        sidx = bank["sidx"][path]
        swgt = bank["swgt"][path]
        total_mass += float(w.sum())
        for s in range(sidx.shape[0]):
            own.index_add_(0, sidx[s].reshape(-1).long(),
                           (swgt[s].float() * w).reshape(-1).double())
    own /= total_mass
    del bank, target_cpu

    spec, scales, dim = load_spec(run_dir, device)
    model = load_stream_model(run_dir / "stream_model.pt", device)
    C = int(model["config"]["C"])
    loader = make_loader(args, cap, 0, 1)
    gen = torch.Generator().manual_seed(args.seed)

    fire = torch.zeros(C, device=device, dtype=torch.float64)
    acc = {t: {"top1": 0.0, "pr": 0.0, "expH": 0.0,
               "thresh": torch.zeros(len(THRESH), device=device,
                                     dtype=torch.float64),
               "cum": torch.zeros(len(RANKS), device=device,
                                  dtype=torch.float64)} for t in TEMPS}
    n = 0
    t0 = time.time()
    for b in range(args.batches):
        idx, pos, bi = sampled_batch(loader, gen, device, args.pos_per_seq)
        phi, _ = pass_features(cfg, cap, idx, pos, bi, spec, scales, dim)
        x = phi.clamp(-6e4, 6e4).half().float()
        y = F.normalize((x - model["mean"]) @ model["projector"], dim=1)
        sims = y @ model["centroids"].t()
        fire += torch.bincount(sims.argmax(1), minlength=C).double()
        n += sims.shape[0]
        for t in TEMPS:
            p = torch.softmax(sims / t, dim=1)
            a = acc[t]
            srt = p.sort(dim=1, descending=True).values
            a["top1"] += srt[:, 0].double().sum().item()
            a["pr"] += (1.0 / (p * p).sum(1)).double().sum().item()
            a["expH"] += torch.exp(
                -(p * (p + 1e-30).log()).sum(1)).double().sum().item()
            for j, th in enumerate(THRESH):
                a["thresh"][j] += (p >= th).sum(1).double().sum()
            cum = srt.cumsum(1)
            for j, r in enumerate(RANKS):
                a["cum"][j] += cum[:, r - 1].double().sum()
        if b % 20 == 0:
            log(f"stats {b}/{args.batches} ({time.time() - t0:.0f}s, "
                f"{n:,} positions)")

    share = (fire / n).cpu()
    out = {
        "positions": n,
        "C": C,
        "cluster_mass": {
            "source": "hard argmax assignment over this sample",
            "mean": float(share.mean()), "median": float(share.median()),
            "min": float(share.min()), "max": float(share.max()),
            "max_over_median": float(share.max() / share.median()),
            "top10pct_share": float(
                share.sort(descending=True).values[:C // 10].sum()),
        },
        "weight_ownership": {
            "note": "fraction of the model's squared weight mass owned by a "
                    "component under the soft partition (soft_s=8)",
            "mean": float(own.mean()), "median": float(own.median()),
            "min": float(own.min()), "max": float(own.max()),
            "top10pct_share": float(
                own.sort(descending=True).values[:C // 10].sum()),
        },
        "posterior": {},
    }
    for t in TEMPS:
        a = acc[t]
        out["posterior"][str(t)] = {
            "mean_top1_mass": a["top1"] / n,
            "participation_ratio": a["pr"] / n,
            "exp_entropy": a["expH"] / n,
            "mean_count_above": {str(th): (a["thresh"][j] / n).item()
                                 for j, th in enumerate(THRESH)},
            "mean_cumulative_mass_at_rank": {
                str(r): (a["cum"][j] / n).item()
                for j, r in enumerate(RANKS)},
        }
    path = run_dir / args.out
    path.write_text(json.dumps(out, indent=1))
    torch.save({"cluster_share": share, "weight_ownership": own.float()},
               run_dir / "decomp_stats_percomponent.pt")
    log(f"stats over {n:,} positions -> {path}")
    log(json.dumps(out["posterior"]["0.05"], indent=1))


if __name__ == "__main__":
    main()
