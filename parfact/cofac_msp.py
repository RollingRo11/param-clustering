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


def forward_signals(model, Ss, probs, n_events, batch, dev, gen):
    """Per-event (delta, h) pairs for both matrices: grad_W s_i = delta h^T."""
    W2 = model.fc2.weight.detach()
    xs, hs, d1s, d2s, tasks_all = [], [], [], [], []
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
        xs.append(x)
        hs.append(a)
        d1s.append(delta1)
        d2s.append(delta2)
        tasks_all.append(tasks.cpu())
    sig = {"fc1.weight": (torch.cat(d1s), torch.cat(xs)),
           "fc2.weight": (torch.cat(d2s), torch.cat(hs))}
    return sig, torch.cat(tasks_all)


def svd_attributions(model, sig):
    """Variant B: A [N, J] over rank-one SVD atoms, sigma-weighted."""
    Ws = {"fc1.weight": model.fc1.weight.detach(),
          "fc2.weight": model.fc2.weight.detach()}
    blocks, atom_matrix, sigmas = [], [], []
    for name in MATRICES:
        U, S, Vh = torch.linalg.svd(Ws[name], full_matrices=False)
        dlt, h = sig[name]
        blocks.append((S[None, :] * (dlt @ U) * (h @ Vh.T)).cpu())
        atom_matrix += [name] * S.numel()
        sigmas.append(S.cpu())
    return torch.cat(blocks, dim=1), atom_matrix, torch.cat(sigmas)


def gradpca_attributions(model, sig, r_per, pca_events, dev, chunk=2048):
    """Variant C (usage-adapted atoms): per matrix, an orthonormal basis
    {d_q} from PCA of the per-event gradient set, plus the residual
    B_res = W - sum_q <W,d_q> d_q as a final atom so reconstruction stays
    exact. Attribution is the bare direction projection <grad_i, d_q>_F.

    The gradients are rank-1 (delta h^T), so the PCA Gram matrix is
    K_ij = (delta_i . delta_j)(h_i . h_j) -- an elementwise product of two
    Gram matrices; eigenvectors of K give the principal directions without
    ever materializing a per-event gradient."""
    Ws = {"fc1.weight": model.fc1.weight.detach(),
          "fc2.weight": model.fc2.weight.detach()}
    blocks, atom_matrix, sigmas = [], [], []
    for name in MATRICES:
        dlt, h = sig[name]
        W = Ws[name]
        R = min(r_per[name], W.shape[0] * W.shape[1] - 1)
        P = min(pca_events, dlt.shape[0])
        dp, hp = dlt[:P], h[:P]
        K = (dp @ dp.T) * (hp @ hp.T)
        evals, evecs = torch.linalg.eigh(K)
        alpha = evecs[:, -R:].flip(-1)                     # top-R [P, R]
        D = torch.einsum("pr,po,pi->roi", alpha, dp, hp)
        Dflat = D.reshape(R, -1)
        # eigen-directions are orthogonal in exact arithmetic; re-orthonormalize
        Q, _ = torch.linalg.qr(Dflat.T)                    # [dim, R]
        Dflat = Q.T
        D = Dflat.reshape(R, *W.shape)
        c = Dflat @ W.reshape(-1)                          # <W, d_q>  [R]
        Bres = W - (c[:, None] * Dflat).sum(0).reshape(W.shape)
        res_norm = Bres.norm().clamp_min(1e-12)
        dres = Bres / res_norm
        A_cols = []
        for i0 in range(0, dlt.shape[0], chunk):
            dc, hc = dlt[i0:i0 + chunk], h[i0:i0 + chunk]
            proj = torch.einsum("bo,roi,bi->br", dc, D, hc)
            pres = torch.einsum("bo,oi,bi->b", dc, dres, hc)
            A_cols.append(torch.cat([proj, pres[:, None]], dim=1).cpu())
        blocks.append(torch.cat(A_cols))
        atom_matrix += [name] * (R + 1)
        sigmas.append(torch.cat([c.abs().cpu(),
                                 res_norm.reshape(1).cpu()]))
        rec = (c[:, None] * Dflat).sum(0).reshape(W.shape) + Bres
        print(f"{name}: R={R} atoms + residual "
              f"(|res| = {float(res_norm):.4f}, "
              f"{float(res_norm**2 / W.pow(2).sum()):.3f} of ||W||^2); "
              f"recon max err {float((rec - W).abs().max()):.2e}", flush=True)
    return torch.cat(blocks, dim=1), atom_matrix, torch.cat(sigmas)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_events", type=int, default=16_384)
    ap.add_argument("--basis", choices=("svd", "gradpca"), default="svd",
                    help="variant B (SVD atoms) or variant C "
                         "(gradient-PCA usage-adapted atoms + residual)")
    ap.add_argument("--r_fc1", type=int, default=100)
    ap.add_argument("--r_fc2", type=int, default=32)
    ap.add_argument("--pca_events", type=int, default=8192)
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

    sig, tasks = forward_signals(model, Ss, probs, args.n_events, 4096, dev,
                                 gen)
    if args.basis == "svd":
        A, atom_matrix, sigma = svd_attributions(model, sig)
    else:
        r_per = {"fc1.weight": args.r_fc1, "fc2.weight": args.r_fc2}
        A, atom_matrix, sigma = gradpca_attributions(model, sig, r_per,
                                                     args.pca_events, dev)
    print(f"A: {tuple(A.shape)} (events x atoms, basis={args.basis}), "
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
