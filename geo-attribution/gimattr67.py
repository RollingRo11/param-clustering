"""Absolute unit membership from attribution (GIM sensor by default).

Components are the geo token clusters (labels in lab{lab_tag}{clu_tag}.pt).
For every unit u (unitabl67.build_units: MLP neuron, QK per head, OV per head
dim, whole head) and cluster c:

    A[c, u] = mean over the cluster's tokens of |a_u(t)|

where a_u(t) is the unit's first-order attribution at token t: the signed sum
over the unit's weight entries of grad x weight x input, i.e. the first-order
estimate of what removing the whole unit does to the summed gold logits.  For
a read-side row r it is g_r(t) (W p(t))_r; for a write-side column j it is
p_j(t) (W^T g(t))_j.  One backward per sequence under the chosen sensor
(sensor_study67.SENSORS; "gim" = corrected backward, K=1).

Unit u belongs to component c iff A[c, u] > tau, tau an absolute threshold in
gold-logit units.  A unit may belong to several components.  Units no cluster
claims at that tau become singleton components (no pool, no always-on set).

Writes own{lab_tag}{clu_tag}_{grouping}_a<tau>.pt for every tau in the grid,
in the layout unitabl67.phase_eval reads, and names the tau to evaluate
(largest tau whose coverage of units is >= --coverage) in --tau_file.

    STREAM67_RUN=$COFAC_DATA/geo67/stream8k \
    python -u gimattr67.py --sensor gim --lab_tag _1m --grouping fine
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import stream67 as S                      # sets CKPT / pile cache, TF32
import sensor_study67 as S67
from sensor_study67 import MODULES, SENSORS, capture
import unitabl67 as U


def log(m):
    print(f"[gimattr] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sensor", default="gim")
    ap.add_argument("--lab_tag", default="_1m")
    ap.add_argument("--clu_tag", default="")
    ap.add_argument("--grouping", choices=list(U.GROUPINGS), default="fine")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--coverage", type=float, default=0.95,
                    help="pick the largest tau at which this fraction of units "
                         "belongs to at least one cluster component")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--out", default="out/gimattr67.json")
    ap.add_argument("--tau_file", default="")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))

    cfg = SENSORS[args.sensor]
    K, path = cfg["K"], cfg.get("ig_path", "weights")
    lab = torch.load(U.lab_path(args), map_location="cpu", weights_only=False)
    ids, labels = lab["ids"].long(), lab["labels"].long()
    C = int(lab["C"])
    N, T = ids.shape
    units = U.build_units()
    n_all = len(units)
    net = U.Net(dev, cfg["mode"])
    looks = {t: U.type_lookup(units, (t,), net.W0, dev, strict=False) for t in U.TYPES}
    log(f"corpus {N}x{T} tokens, C={C} clusters, {n_all} units, sensor {args.sensor} "
        f"(mode {cfg['mode']}, K={K})")

    A_abs = torch.zeros(C, n_all, device=dev)
    A_sgn = torch.zeros(C, n_all, device=dev)
    cnt = torch.zeros(C, device=dev)
    tot_abs = torch.zeros(n_all, device=dev, dtype=torch.float64)
    n_tok = 0
    t0 = time.perf_counter()
    for b0 in range(0, N, args.batch):
        idx = ids[b0:b0 + args.batch].to(dev)
        lb = labels[b0:b0 + args.batch].to(dev).flatten()
        B = idx.shape[0]
        a_units = torch.zeros(B * T, n_all, device=dev)
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for p in MODULES:
                    net.Wt[p].copy_(net.W0[p] * (a if path == "weights" else 1.0))
            P, G = capture(net.model, idx, scalar=cfg.get("scalar", "gold"))
            with torch.no_grad():
                for p in MODULES:
                    W = net.Wt[p]
                    if S.is_write_side(p):
                        side = P[p].float() * (G[p].float() @ W)            # [B,T,d_in]
                    else:
                        side = G[p].float() * (P[p].float() @ W.t())        # [B,T,d_out]
                    side = side.reshape(B * T, -1) / K
                    for lk in looks.values():
                        u_ids = lk[p]
                        ok = u_ids >= 0
                        if ok.any():
                            a_units.index_add_(1, u_ids[ok], side[:, ok])
            del P, G
        net.restore()
        with torch.no_grad():
            valid = lb >= 0
            av, lv = a_units[valid], lb[valid]
            A_abs.index_add_(0, lv, av.abs())
            A_sgn.index_add_(0, lv, av)
            cnt.index_add_(0, lv, torch.ones(int(valid.sum()), device=dev))
            tot_abs += av.abs().sum(0).double()
            n_tok += int(valid.sum())
        del a_units
        if (b0 // args.batch) % 32 == 0:
            log(f"{b0 + B}/{N} seqs ({time.perf_counter() - t0:.0f}s)")
    log(f"attribution done: {n_tok} tokens ({time.perf_counter() - t0:.0f}s)")

    A = (A_abs / cnt.clamp_min(1)[:, None]).cpu()
    A_s = (A_sgn / cnt.clamp_min(1)[:, None]).cpu()
    unit_mean_abs = (tot_abs / max(n_tok, 1)).float().cpu()
    cnt = cnt.cpu()
    stats_path = U.out_dir(args) / f"gimattr{args.lab_tag}{args.clu_tag}_{args.sensor}.pt"
    torch.save({"A_abs_mean": A, "A_signed_mean": A_s, "counts": cnt,
                "unit_mean_abs": unit_mean_abs, "C": C, "sensor": args.sensor,
                "n_tok": n_tok, "unit_type": [u[0] for u in units]}, stats_path)
    log(f"wrote {stats_path.name}")

    sub = U.unit_ids_of(units, args.grouping)
    Asub = A[:, sub]
    covered = cnt >= 1
    summary = {"format": "gimattr67_v1", "sensor": args.sensor, "lab_tag": args.lab_tag,
               "clu_tag": args.clu_tag, "grouping": args.grouping, "C": C, "n_tok": n_tok,
               "n_units": int(sub.numel()), "clusters_with_tokens": int(covered.sum()),
               "unit_mean_abs_attr_quantiles": U.qdict(unit_mean_abs[sub].numpy(), (0.1, 0.5, 0.9, 0.99)),
               "A_quantiles_over_covered_cells": U.qdict(Asub[covered].numpy().ravel(), (0.5, 0.9, 0.99, 0.999)),
               "per_tau": {}}
    chosen = None
    for tau in args.taus:
        M = (Asub > tau) & covered[:, None]                  # [C, n_sub]
        per_unit = M.sum(0)
        sizes = M.sum(1)
        coverage = float((per_unit > 0).float().mean())
        unowned = sub[per_unit == 0]
        members = [sub[M[c]] for c in range(C)] + [torch.tensor([int(g)]) for g in unowned.tolist()]
        variant = f"a{tau:g}"
        own = {"grouping": args.grouping, "variant": variant,
               "C": C + int(unowned.numel()), "n_clusters": C,
               "n_singletons": int(unowned.numel()),
               "members": members, "pool": [],
               "inert": torch.zeros(0, dtype=torch.long),
               "unit_ids": sub, "unit_mean_kl": unit_mean_abs[sub],
               "tau": tau, "sensor": args.sensor, "rule": "abs attribution mean > tau",
               "covered": covered.nonzero().flatten(), "counts": cnt}
        torch.save(own, U.out_dir(args) / f"own{args.lab_tag}{args.clu_tag}_{args.grouping}_{variant}.pt")
        st = {"coverage": round(coverage, 4),
              "memberships": int(M.sum()),
              "components_per_unit_quantiles": U.qdict(per_unit.float().numpy(), (0.5, 0.9, 0.99)),
              "mean_components_per_unit": round(float(per_unit.float().mean()), 2),
              "units_per_cluster_quantiles": U.qdict(sizes[covered].float().numpy(), (0.1, 0.5, 0.9, 0.99)),
              "largest_cluster_component": int(sizes.max()),
              "empty_cluster_components": int((sizes[covered] == 0).sum()),
              "singletons": int(unowned.numel())}
        summary["per_tau"][f"{tau:g}"] = st
        log(f"tau {tau:g}: coverage {coverage:.3f}, comps/unit mean {st['mean_components_per_unit']}, "
            f"units/cluster median {st['units_per_cluster_quantiles']['0.5']:.0f} max {st['largest_cluster_component']}, "
            f"singletons {st['singletons']}")
        if coverage >= args.coverage:
            chosen = variant
    summary["chosen_variant"] = chosen
    log(f"chosen variant (largest tau with coverage >= {args.coverage}): {chosen}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    if args.tau_file and chosen:
        Path(args.tau_file).write_text(f"{args.grouping}_{chosen}\n")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
