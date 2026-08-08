"""Ablation-robustness curve: CE as a function of how many components are removed.

This is VPD Section 3.2's plot, but it measures a different thing here and the
difference is worth stating rather than glossing.

VPD's decomposition APPROXIMATES the target's weights, so it needs a residual
Delta-component and its curve is partly a FAITHFULNESS curve — it reports CE
2.72-3.02 against the target's 2.71, and the gap is reconstruction error. Our
shares sum to exactly 1 per weight ENTRY, so ablating zero components
reproduces the target bit-for-bit and ablating all of them zeroes every matrix.
There is no reconstruction error to plot. What is left is a pure MINIMALITY
curve: how much of the decomposition can be deleted before behaviour goes.

Only the left endpoint is known in advance: at K = 0 the model is bit-for-bit
the target. The right endpoint is NOT uniform — a first version of this script
claimed it was and that was wrong. Ablating all C components zeroes the 112
transformer matrices but leaves the embedding, the final RMSNorm and the
unembedding, so the model collapses to the direct embed -> unembed path, which
is confidently WRONG rather than uninformed: CE ~90 nats, far above ln(128256)
= 11.76. Anything above 11.76 on this plot is a model doing worse than guessing
uniformly.

Speed: ablating a SET S is just W * (1 - sum_{c in S} s_c), one elementwise
mask per matrix, so nothing is materialised per component. The whole bank is
int16 + fp16 and comes to 31 GB, which fits on one B200, so it is loaded once
and every K is a pass over resident memory. The per-component ranks are gathered
once per ordering and reused for all K.

    python3.12 ablation_curve.py --orders random mass_asc mass_desc
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from german_vpd_1b import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--n_blocks", type=int, default=48,
                    help="the first 8 Pile blocks are anomalously "
                         "predictable (CE 0.45 vs 1.24 over all 24), "
                         "so the default takes every eval block")
    ap.add_argument("--orders", nargs="+",
                    default=["random", "mass_asc", "mass_desc"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ablation_curve.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    idx = torch.cat([data["pile_eval"],
                     data["bio_retain_eval"]])[:args.n_blocks].to(dev)
    target = geo1b.load_target_1b(dev)

    @torch.no_grad()
    def ce():
        out = []
        for s in range(0, idx.shape[0], 4):
            b = idx[s:s + 4]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                lg = target(b)
            out.append(F.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]).float(),
                b[:, 1:].reshape(-1)))
        return float(torch.stack(out).mean())

    t0 = time.perf_counter()
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt", weights_only=True,
                      map_location="cpu", mmap=True)
    C = int(bank["C"])
    mods = list(bank["modules"])
    uniform = math.log(128256)

    # ---- one pass: pull sidx/swgt onto the GPU, keep originals, get masses ----
    S, Wt, W0 = {}, {}, {}
    mass = torch.zeros(C, device=dev, dtype=torch.float64)
    for p in mods:
        sidx = bank["sidx"][p].to(dev, non_blocking=True)
        swgt = bank["swgt"][p].to(dev, non_blocking=True)
        w = target.get_submodule(p).weight
        W0[p] = w.detach().clone()
        Wt[p] = w
        contrib = (swgt.float() * (W0[p] ** 2)[None])
        mass += torch.bincount(sidx.reshape(-1).int(),
                               weights=contrib.reshape(-1).double(),
                               minlength=C)
        S[p] = (sidx, swgt)
        del contrib
    base_ce = ce()
    log(f"bank resident on GPU in {time.perf_counter() - t0:.0f}s; "
        f"base CE {base_ce:.4f} on {idx.shape[0]} blocks; uniform {uniform:.4f}; "
        f"{torch.cuda.memory_allocated(dev) / 2**30:.0f} GiB allocated")

    KS = sorted({0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
                 512, 768, 1024, 1536, 2048, 2560, 3072, 3456, 3712, 3840,
                 3968, 4032, 4064, 4080, 4096, C})
    orders = {}
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    if "random" in args.orders:
        orders["random"] = torch.randperm(C, generator=g)
    if "mass_asc" in args.orders:
        orders["mass_asc"] = torch.argsort(mass).cpu()
    if "mass_desc" in args.orders:
        orders["mass_desc"] = torch.argsort(mass, descending=True).cpu()

    results = {}
    for name, order in orders.items():
        # rank[c] = position of c in this ablation order, gathered once
        rank = torch.empty(C, dtype=torch.int32, device=dev)
        rank[order.to(dev)] = torch.arange(C, dtype=torch.int32, device=dev)
        R = {p: rank[S[p][0].int()] for p in mods}
        curve, t1 = [], time.perf_counter()
        for K in KS:
            with torch.no_grad():
                for p in mods:
                    if K == 0:
                        Wt[p].copy_(W0[p])
                        continue
                    keep = (S[p][1] * (R[p] < K)).sum(0, dtype=torch.float32)
                    Wt[p].copy_(W0[p] * (1.0 - keep))
            v = ce()
            curve.append({"k": K, "frac": round(K / C, 5), "ce": round(v, 5),
                          "delta": round(v - base_ce, 5)})
            log(f"  {name:<10} K={K:<5} ({100 * K / C:5.1f}%)  "
                f"CE {v:8.4f}  Δ {v - base_ce:+8.4f}")
        del R
        torch.cuda.empty_cache()
        results[name] = curve
        log(f"{name}: {len(KS)} points in {time.perf_counter() - t1:.0f}s")

    with torch.no_grad():
        for p in mods:
            Wt[p].copy_(W0[p])

    # how many can go before CE moves by a given amount?
    def budget(curve, tol):
        ok = [r["k"] for r in curve if r["delta"] <= tol]
        return max(ok) if ok else 0

    out = {"format": "ablation_curve_v1", "C": C, "base_ce": round(base_ce, 5),
           "uniform_ce": round(uniform, 5), "n_blocks": args.n_blocks,
           "ks": KS, "curves": results,
           "components_removable_within": {
               name: {f"{t}": budget(c, t) for t in (0.01, 0.05, 0.1, 0.5)}
               for name, c in results.items()}}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    for name, b in out["components_removable_within"].items():
        log(f"{name:<10} removable within ΔCE: " +
            "  ".join(f"{t}: {v} ({100 * v / C:.0f}%)" for t, v in b.items()))
    log(f"wrote {run_dir / args.out}  (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
