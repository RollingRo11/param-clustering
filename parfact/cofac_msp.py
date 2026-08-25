"""Co-factorization (v2, U-simplex + residual V + I-divergence) for the
multitask sparse parity MLP (msp_model.py checkpoint).

Events are single inputs sampled from the training (power-law) task
distribution; the score is the logit margin s = z[y*] - z[other] with
y* = argmax. Atoms are variant-B rank-one SVD atoms of fc1.weight and
fc2.weight. For this depth-2 MLP the per-event weight gradients are closed
form (no autograd):

    grad_W2 s = delta2 a^T,  delta2 = e_{y*} - e_{other}
    grad_W1 s = delta1 x^T,  delta1 = (W2^T delta2) * 1[z1 > 0]

so atom attributions are a_iq = sigma_q (u_q . delta)(v_q . h_in), batched.

    python cofac_msp.py --ckpt out/msp/model.pt --out out/msp/cofac_C300
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from msp_model import MSPModel, sample_batch, N_TASKS, N_BITS
from cofact_v2 import TriFactorizationV2, component_mass

MATRICES = ("fc1.weight", "fc2.weight")


def collect(model, Ss, probs, n_events, batch, dev, gen):
    """Signed atom-attribution matrix A [N, J], task ids, and SVD bases."""
    W1, W2 = model.fc1.weight.detach(), model.fc2.weight.detach()
    svd = {}
    for name, W in zip(MATRICES, (W1, W2)):
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        svd[name] = (U, S, Vh)                        # W = U diag(S) Vh
    A_rows, tasks_all = [], []
    for i0 in range(0, n_events, batch):
        b = min(batch, n_events - i0)
        x, _, tasks = sample_batch(b, Ss, probs, N_TASKS, N_BITS, dev, gen)
        with torch.no_grad():
            z1 = model.fc1(x)
            a = F.relu(z1)
            z2 = model.fc2(a)
            ystar = z2.argmax(-1)
            delta2 = F.one_hot(ystar, 2).float() - F.one_hot(1 - ystar,
                                                             2).float()
            delta1 = (delta2 @ W2) * (z1 > 0).float()
            blocks = []
            for name, (dlt, h) in zip(MATRICES,
                                      ((delta1, x), (delta2, a))):
                U, S, Vh = svd[name]
                blocks.append(S[None, :] * (dlt @ U) * (h @ Vh.T))
            A_rows.append(torch.cat(blocks, dim=1).cpu())
            tasks_all.append(tasks.cpu())
    A = torch.cat(A_rows)
    atom_matrix = [n for n in MATRICES for _ in range(svd[n][1].numel())]
    sigma = torch.cat([svd[n][1] for n in MATRICES]).cpu()
    return A, torch.cat(tasks_all), atom_matrix, sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_events", type=int, default=16_384)
    ap.add_argument("--k_factors", type=int, default=600)
    ap.add_argument("--c_groups", type=int, default=300)
    ap.add_argument("--fact_steps", type=int, default=4000)
    ap.add_argument("--no_u_simplex", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/msp/model.pt")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "out/msp/cofac_C300")
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = MSPModel(width=int(ck["config"]["width"])).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    Ss = [torch.tensor(s) for s in ck["Ss"]]
    probs = torch.tensor(ck["probs"])
    gen = torch.Generator(device=dev).manual_seed(args.seed + 1000)

    A, tasks, atom_matrix, sigma = collect(model, Ss, probs, args.n_events,
                                           4096, dev, gen)
    print(f"A: {tuple(A.shape)} (events x atoms), "
          f"atoms per matrix: fc1={atom_matrix.count('fc1.weight')} "
          f"fc2={atom_matrix.count('fc2.weight')}", flush=True)

    # per-matrix RMS normalization, then row-normalized magnitudes
    An = A.clone()
    for name in MATRICES:
        cols = [j for j, m in enumerate(atom_matrix) if m == name]
        rms = An[:, cols].pow(2).mean().sqrt().clamp_min(1e-12)
        An[:, cols] /= rms
    M = An.abs()
    M_bar = M / M.sum(dim=1, keepdim=True).clamp_min(1e-8)

    torch.manual_seed(args.seed)
    fact = TriFactorizationV2(args.n_events, A.shape[1], args.k_factors,
                              args.c_groups,
                              u_simplex=not args.no_u_simplex,
                              seed=args.seed).to(dev)
    met = fact.fit(M_bar.to(dev), steps=args.fact_steps)

    with torch.no_grad():
        V = fact.V.detach().cpu()
        r = fact.residual.detach().cpu()
        z = An @ V                                    # signed usage [N, C]
        # mean |usage| per (task, component) -- the task-selectivity table
        task_usage = torch.zeros(N_TASKS, args.c_groups)
        counts = torch.zeros(N_TASKS)
        for t in range(N_TASKS):
            sel = tasks == t
            counts[t] = sel.sum()
            if sel.any():
                task_usage[t] = z[sel].abs().mean(0)

    mass = component_mass(V, r, sigma)
    met["mass"] = mass
    met["u_simplex"] = not args.no_u_simplex
    met["config"] = {k: str(v) for k, v in vars(args).items()}
    met["events_per_task_head"] = counts[:10].tolist()

    torch.save({"U": fact.U.detach().cpu(), "S": fact.S.detach().cpu(),
                "V": V, "r": r, "sigma": sigma, "atom_matrix": atom_matrix,
                "A": A.half(), "tasks": tasks, "z": z.half(),
                "task_usage": task_usage,
                "config": met["config"]}, args.out / "factorization.pt")
    (args.out / "metrics.json").write_text(json.dumps(
        {k: v for k, v in met.items()}, indent=1))
    print("component mass: n50", mass["components_for_50pct_mass"],
          "n90", mass["components_for_90pct_mass"],
          "top fro_share", mass["per_component"][0]["fro_share"])
    print("saved", args.out)

    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            run = wandb.init(project=os.environ.get("WANDB_PROJECT",
                                                    "param-clustering"),
                             id=f"{args.out.parent.name}-{args.out.name}",
                             name=f"{args.out.parent.name}-{args.out.name}",
                             resume="allow",
                             dir=str(args.out), config=met["config"])
            run.log({"idiv": met["idiv"], "rel_err": met["rel_err"],
                     "r2_attr_euclid": met["r2_attr_euclid"],
                     "n50": mass["components_for_50pct_mass"],
                     "n90": mass["components_for_90pct_mass"]})
            art = wandb.Artifact(f"{args.out.parent.name}-{args.out.name}",
                                 type="cofac-decomposition")
            art.add_file(str(args.out / "factorization.pt"))
            art.add_file(str(args.out / "metrics.json"))
            run.log_artifact(art)
            run.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
