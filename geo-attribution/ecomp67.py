"""Components straight from the measured unit-ablation effect matrix E.

E[u, x] = KL(full || unit u zeroed) at token x, as written by
unitabl67.py effects.  Units are whole MLP neurons, QK per head, and OV per
head dim (unitabl67.build_units, grouping "fine"), so removing a component
removes whole units and nothing is scaled fractionally.

One step: a unit's row of E is its effect profile over the corpus tokens.
Rows are random-projected to `proj_dim` dims (dot products are preserved in
expectation), L2-normalised, and split into C groups by spherical k-means.
Every unit lands in exactly one component.  No pool, no residual, no
always-on set: whatever is always on just clusters with whatever else is.

Writes own{lab_tag}_{grouping}_e{C}.pt in the layout unitabl67.phase_eval
reads, so the keep / drop curves come from the existing eval unchanged:

    STREAM67_RUN=$COFAC_DATA/geo67/stream8k \
    python -u ecomp67.py --lab_tag _1m --nshard 6 --nq 16 --grouping fine --C 1024
    python -u unitabl67.py eval --lab_tag _1m --variant fine_e1024 --sel_comps 0 ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import stream67 as S                      # sets CKPT / pile cache, TF32
import sensor_study67 as S67
import unitabl67 as U


def log(m):
    print(f"[ecomp] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lab_tag", default="_1m", help="corpus tag of the effect matrix")
    ap.add_argument("--clu_tag", default="")
    ap.add_argument("--nshard", type=int, default=6)
    ap.add_argument("--nq", type=int, default=16)
    ap.add_argument("--grouping", choices=list(U.GROUPINGS), default="fine")
    ap.add_argument("--C", type=int, nargs="+", default=[1024])
    ap.add_argument("--proj_dim", type=int, default=1024)
    ap.add_argument("--row_chunk", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["E", "attr"], default="E",
                    help="unit profiles: rows of the measured effect matrix E "
                         "(random-projected), or the cluster-mean |attribution| "
                         "table written by gimattr67 (one dim per token cluster)")
    ap.add_argument("--attr_sensor", default="gim")
    ap.add_argument("--min_count", type=int, default=4,
                    help="attr: token clusters with fewer tokens are dropped from the profile")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--out", default="out/ecomp67.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))

    units = U.build_units()
    n_all = len(units)
    sub = U.unit_ids_of(units, args.grouping)
    t0 = time.perf_counter()
    if args.source == "attr":
        st = torch.load(U.out_dir(args) / f"gimattr{args.lab_tag}{args.clu_tag}_{args.attr_sensor}.pt",
                        map_location="cpu", weights_only=False)
        ok = st["counts"] >= args.min_count
        Y = st["A_abs_mean"][ok].t().contiguous().to(dev)      # [n_all, n_clusters_used]
        mean_kl = st["unit_mean_abs"].float()
        n_tok = int(st["n_tok"])
        prefix = "c"
        log(f"attribution profiles ({args.attr_sensor}): {n_all} units x {int(ok.sum())} token "
            f"clusters with >= {args.min_count} tokens; grouping {args.grouping} = {sub.numel()} units")
    else:
        lab, parts = U.load_effects(args, "kl")
        N, T = lab["ids"].shape
        valid = (lab["labels"].flatten() >= 0)
        valid_idx = valid.nonzero().flatten().to(dev)
        n_tok = int(valid_idx.numel())
        prefix = "e"
        log(f"E{args.lab_tag}: {n_all} unit rows x {N * T} tokens ({n_tok} valid); "
            f"grouping {args.grouping} = {sub.numel()} units")

        # random projection of every unit's effect profile: [n_all, proj_dim]
        g = torch.Generator(device=dev).manual_seed(args.seed)
        R = torch.randn(n_tok, args.proj_dim, generator=g, device=dev) / math.sqrt(args.proj_dim)
        Y = torch.zeros(n_all, args.proj_dim, device=dev)
        mean_kl = torch.zeros(n_all)
        for g0 in range(0, n_all, args.row_chunk):
            g1 = min(n_all, g0 + args.row_chunk)
            X = torch.from_numpy(U.read_rows(parts, g0, g1)).to(dev)[:, valid_idx].float()
            mean_kl[g0:g1] = X.mean(1).cpu()
            Y[g0:g1] = X @ R
            del X
            if (g1 // args.row_chunk) % 8 == 0:
                log(f"projected {g1}/{n_all} units ({time.perf_counter() - t0:.0f}s)")
        del R
        torch.cuda.empty_cache()
        log(f"projection done ({time.perf_counter() - t0:.0f}s)")

    Z = Y[sub.to(dev)]
    zero = Z.norm(dim=1) == 0                     # units with no effect anywhere
    log(f"{int(zero.sum())} units have an all-zero profile")
    summary = {"format": "ecomp67_v1", "source": args.source, "lab_tag": args.lab_tag,
               "grouping": args.grouping, "profile_dim": int(Y.shape[1]),
               "n_units": int(sub.numel()), "n_tok": n_tok, "proj_dim": args.proj_dim,
               "seed": args.seed, "zero_profile_units": int(zero.sum()), "per_C": {}}
    types = np.array([units[i][0] for i in sub.tolist()])
    for C in args.C:
        t1 = time.perf_counter()
        _, lab_u = S67.kmeans(Z, C, args.kmeans_iters, args.seed, spherical=True)
        lab_u = lab_u.cpu()
        members = [sub[lab_u == c] for c in range(C)]
        sizes = torch.tensor([m.numel() for m in members])
        own = {"grouping": args.grouping, "variant": f"{prefix}{C}", "C": C, "source": args.source,
               "members": members, "pool": [],
               "inert": torch.zeros(0, dtype=torch.long),
               "unit_ids": sub, "unit_mean_kl": mean_kl[sub],
               "group_label": lab_u, "proj_dim": args.proj_dim, "seed": args.seed,
               "lab_tag": args.lab_tag, "n_tok": n_tok}
        path = U.out_dir(args) / f"own{args.lab_tag}{args.clu_tag}_{args.grouping}_{prefix}{C}.pt"
        torch.save(own, path)
        mixed = sum(1 for c in range(C) if sizes[c] > 0
                    and len(set(types[(lab_u == c).numpy()])) > 1)
        summary["per_C"][str(C)] = {
            "sizes_quantiles": U.qdict(sizes[sizes > 0].numpy(), (0.1, 0.5, 0.9, 0.99)),
            "largest": int(sizes.max()), "empty": int((sizes == 0).sum()),
            "components_mixing_unit_types": mixed,
            "unit_mean_kl_quantiles": U.qdict(mean_kl[sub].numpy())}
        log(f"C={C}: sizes median {int(sizes.median())} max {int(sizes.max())} "
            f"empty {int((sizes == 0).sum())}, {mixed} components mix unit types "
            f"({time.perf_counter() - t1:.0f}s) -> {path.name}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
