"""Task-mechanism matching for the MSP decompositions.

Implements the VPD paper's Appendix A.8 clustering (MDL cost, stochastic
hierarchical merging: binarize subcomponent causal importances at tau, build
the coactivation matrix, merge pairs sampled by exp(-gamma * rank of
delta-MDL), stop where the marginal delta-MDL crosses 0) on our SPD run's
gate importances, then scores every unit system on the same question:

    does each learned task get ONE mechanism, and does that mechanism
    belong to that task?

Unit systems scored: co-fac components (any C), SPD raw subcomponents, and
SPD MDL-clusters. For each system we build a task-usage table
mean-usage[task, unit] (|z| for co-fac, gate CI for SPD; cluster usage =
max over members), then per learned task find its best unit and report
  coverage = usage(task, best) / sum_units usage(task, .)
  purity   = usage(task, best) / sum_tasks usage(., best)
  F1       = harmonic mean, plus how many tasks share the same best unit.

    python msp_mechmatch.py --model out/msp_w1000/model.pt \
        --cofac out/msp_w1000/cofac_C300 out/msp_w1000/cofac_C100 \
        --spd out/msp_w1000/spd_C300 --out out/msp_w1000/mechmatch.json
"""
import argparse
import json
import math
from pathlib import Path

import torch

from msp_model import MSPModel, sample_batch, N_TASKS, N_BITS
from spd_msp import MODULES, MatrixGate, install


# ---------------- VPD appendix A.8 clustering ----------------

def mdl_cluster(act: torch.Tensor, alpha: float = 1.0, gamma: float = 0.2,
                seed: int = 0, log_every: int = 50):
    """act: [B, n] binarized causal importances. Returns (assignment [n],
    n_clusters, trajectory list of (iteration, delta_mdl, k)).

    Vectorized: group activations A [B,k] and their coactivation matrix
    S = A^T A are maintained incrementally, so each merge costs O(k^2)
    scalar work + one k x B OR, not O(k^2 B)."""
    B, n = act.shape
    A = act.clone().float()                                # [B, k]
    S = A.T @ A                                            # coact counts
    s = S.diagonal().clone()                               # [k]
    rank = torch.ones(n)
    members = [[i] for i in range(n)]
    g = torch.Generator().manual_seed(seed)
    traj, snapshots = [], []

    it = 0
    while A.shape[1] > 1:
        k = A.shape[1]
        s_sum = float(s.sum())
        # merged group's activation count for every pair: s_i + s_j - coact
        s_merged = s[:, None] + s[None, :] - S
        rsum = rank[:, None] + rank[None, :]
        dict_red = (s_sum - s[:, None] - s[None, :]) * (
            math.log2(k - 1) - math.log2(k)) if k > 1 else 0
        idx_enc = (s_merged * math.log2(max(k - 1, 1))
                   - (s[:, None] + s[None, :]) * math.log2(k))
        rank_pen = alpha * (s_merged * rsum
                            - (s * rank)[:, None] - (s * rank)[None, :])
        cost = dict_red + idx_enc + rank_pen
        iu = torch.triu_indices(k, k, offset=1)
        flat = cost[iu[0], iu[1]]
        order = flat.argsort()
        N = order.numel()
        u = torch.rand((), generator=g).item()
        J = int(math.floor(-math.log(1 - u * (1 - math.exp(-gamma * N)))
                           / gamma))
        J = min(max(J, 0), N - 1)
        pick = order[J]
        i, j = int(iu[0, pick]), int(iu[1, pick])
        dL = float(flat[pick])
        traj.append((it, dL, k))
        snapshots.append([list(m) for m in members])

        # merge j into i, drop j
        new_col = ((A[:, i] + A[:, j]) > 0).float()
        keep = [q for q in range(k) if q != j]
        A[:, i] = new_col
        A = A[:, keep]
        new_row = new_col @ A                              # [k-1]
        S = S[keep][:, keep]
        pos = keep.index(i)
        S[pos, :] = new_row
        S[:, pos] = new_row
        s = S.diagonal().clone()
        rank[i] = rank[i] + rank[j]
        rank = rank[keep]
        members[i] = members[i] + members[j]
        members.pop(j)
        if it % log_every == 0:
            print(f"  merge {it}: k={k} dL={dL:.1f}", flush=True)
        it += 1

    # stop where marginal delta-MDL crosses 0: last merge with dL < 0
    stop = 0
    for q, (_, dL, _) in enumerate(traj):
        if dL < 0:
            stop = q + 1
    chosen = snapshots[stop] if stop < len(snapshots) else snapshots[-1]
    assign = torch.full((n,), -1, dtype=torch.long)
    for ci, mem in enumerate(chosen):
        for m in mem:
            assign[m] = ci
    print(f"MDL clustering: stopped at merge {stop}, "
          f"{int(assign.max()) + 1} clusters (alpha={alpha})", flush=True)
    return assign, int(assign.max()) + 1, traj


# ---------------- usage tables ----------------

def spd_usage(spd_dir: Path, model_ck, dev, n_events, seed):
    """Mean gate CI per (task, subcomponent): [N_TASKS, 2*C], plus the
    binarized activation matrix [n_events, 2*C] for clustering."""
    state = torch.load(spd_dir / "spd_state.pt", map_location=dev,
                       weights_only=True)
    c_per = int(state["c_per_module"])
    model = MSPModel(width=int(model_ck["config"]["width"])).to(dev)
    model.load_state_dict(model_ck["state_dict"])
    model.eval()
    wrappers = install(model, c_per)
    model.to(dev)
    for n, w in wrappers.items():
        w.V.data.copy_(state["wrappers"][n]["V"].to(dev))
        w.U.data.copy_(state["wrappers"][n]["U"].to(dev))
    d_ins = {"fc1": N_TASKS + N_BITS, "fc2": int(model_ck["config"]["width"])}
    gates = torch.nn.ModuleDict({n: MatrixGate(c_per, d_ins[n])
                                 for n in MODULES}).to(dev)
    gates.load_state_dict(state["gates"])
    Ss = [torch.tensor(x) for x in model_ck["Ss"]]
    probs = torch.tensor(model_ck["probs"])
    gen = torch.Generator(device=dev).manual_seed(seed + 2000)
    gs, ts = [], []
    with torch.no_grad():
        for i0 in range(0, n_events, 4096):
            b = min(4096, n_events - i0)
            x, _, tasks = sample_batch(b, Ss, probs, N_TASKS, N_BITS, dev,
                                       gen)
            for w in wrappers.values():
                w.mode, w.mask = "target", None
            model(x)
            g = torch.cat([gates[n](wrappers[n].last_input)[0]
                           for n in MODULES], dim=1)     # [b, 2*C]
            gs.append(g.cpu())
            ts.append(tasks.cpu())
    g = torch.cat(gs)
    tasks = torch.cat(ts)
    usage = torch.zeros(N_TASKS, g.shape[1])
    for t in range(N_TASKS):
        sel = tasks == t
        if sel.any():
            usage[t] = g[sel].mean(0)
    return usage, g, tasks


def cofac_usage(cofac_dir: Path):
    d = torch.load(cofac_dir / "factorization.pt", map_location="cpu",
                   weights_only=False)
    return d["task_usage"].float()                       # [N_TASKS, C]


# ---------------- the shared score ----------------

def score(usage: torch.Tensor, learned: torch.Tensor):
    """usage [N_TASKS, U]; learned: bool mask of tasks to score."""
    eps = 1e-12
    per_task = []
    best_units = []
    for t in torch.nonzero(learned).squeeze(-1).tolist():
        u = int(usage[t].argmax())
        cov = float(usage[t, u] / (usage[t].sum() + eps))
        pur = float(usage[t, u] / (usage[:, u].sum() + eps))
        f1 = 2 * cov * pur / (cov + pur + eps)
        per_task.append({"task": t, "best_unit": u, "coverage": round(cov, 4),
                         "purity": round(pur, 4), "f1": round(f1, 4)})
        best_units.append(u)
    n_shared = len(best_units) - len(set(best_units))
    f1s = [p["f1"] for p in per_task]
    return {"n_tasks_scored": len(per_task),
            "mean_f1": round(sum(f1s) / max(len(f1s), 1), 4),
            "median_f1": round(sorted(f1s)[len(f1s) // 2], 4),
            "n_units": usage.shape[1],
            "n_distinct_best_units": len(set(best_units)),
            "n_tasks_sharing_a_unit": n_shared,
            "per_task": per_task}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cofac", type=Path, nargs="*", default=[])
    ap.add_argument("--spd", type=Path, default=None)
    ap.add_argument("--n_events", type=int, default=16_384)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.2)
    ap.add_argument("--acc_threshold", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    learned = torch.tensor(ck["per_task_acc"]) > args.acc_threshold
    print(f"scoring {int(learned.sum())}/{N_TASKS} learned tasks", flush=True)
    report = {"learned_tasks": int(learned.sum()),
              "acc_threshold": args.acc_threshold,
              "config": {k: str(v) for k, v in vars(args).items()}}

    for cdir in args.cofac:
        u = cofac_usage(cdir)
        report[f"cofac_{cdir.name}"] = score(u, learned)
        print(cdir.name, {k: v for k, v in report[f"cofac_{cdir.name}"].items()
                          if k != "per_task"}, flush=True)

    if args.spd is not None:
        usage, g, tasks = spd_usage(args.spd, ck, dev, args.n_events,
                                    args.seed)
        report["spd_raw"] = score(usage, learned)
        print("spd_raw", {k: v for k, v in report["spd_raw"].items()
                          if k != "per_task"}, flush=True)

        act = (g > args.tau)
        alive = act.any(0)
        print(f"clustering {int(alive.sum())} alive subcomponents "
              f"(of {g.shape[1]}) at tau={args.tau}", flush=True)
        if int(alive.sum()) < 2:
            report["spd_clustered"] = {"error": "fewer than 2 alive "
                                       "subcomponents at tau"}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report))
            print("wrote", args.out)
            return
        assign_alive, n_cl, traj = mdl_cluster(act[:, alive].contiguous(),
                                               alpha=args.alpha,
                                               gamma=args.gamma,
                                               seed=args.seed)
        assign = torch.full((g.shape[1],), -1, dtype=torch.long)
        assign[alive] = assign_alive
        cl_usage = torch.zeros(N_TASKS, n_cl)
        for c in range(n_cl):
            mem = assign == c
            if mem.any():
                cl_usage[:, c] = usage[:, mem].max(dim=1).values
        report["spd_clustered"] = score(cl_usage, learned)
        report["spd_clustered"]["n_clusters"] = n_cl
        report["spd_clustered"]["n_alive_subcomponents"] = int(alive.sum())
        report["spd_cluster_assignment"] = assign.tolist()
        print("spd_clustered", {k: v for k, v in
                                report["spd_clustered"].items()
                                if k not in ("per_task",)}, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
