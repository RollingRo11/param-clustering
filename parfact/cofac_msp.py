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


def scalar_attributions(model, sig, dev, chunk=2048):
    """Variant A: every scalar weight is its own atom; attribution is
    gradient-times-parameter, a_ij = theta_j * (grad_W s_i)_j
    = W[o,i] * delta[b,o] * h[b,i]. A is [N, 152k] fp16 on device."""
    Ws = {"fc1.weight": model.fc1.weight.detach(),
          "fc2.weight": model.fc2.weight.detach()}
    blocks, atom_matrix, sigmas = [], [], []
    for name in MATRICES:
        dlt, h = sig[name]
        W = Ws[name]
        cols = []
        for i0 in range(0, dlt.shape[0], chunk):
            g = torch.einsum("bo,bi->boi", dlt[i0:i0 + chunk],
                             h[i0:i0 + chunk]) * W[None]
            cols.append(g.reshape(g.shape[0], -1).half())
        blocks.append(torch.cat(cols))
        atom_matrix += [name] * W.numel()
        sigmas.append(W.abs().reshape(-1).cpu())
    return torch.cat(blocks, dim=1).to(dev), atom_matrix, torch.cat(sigmas)


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


def fit_pinned(M_bar, k_factors, c_groups, steps=4000, lr=2e-2, seed=0,
               f_cap=0.9, row_chunk=0, log_every=250):
    """Port of cofac67's fit_pinned (v3b): the shared backbone is a rank-1
    LS fit (per-event coefficient a on the mean direction, per-atom loading
    beta), each atom's backbone fraction f is PINNED (capped at f_cap), and
    V distributes only the remaining (1-f) of each atom over components.
    Drain is impossible; no component can become the backbone."""
    dev = M_bar.device
    n, j = M_bar.shape
    with torch.no_grad():
        mu = M_bar.mean(0)
        u = mu / mu.norm().clamp_min(1e-12)
        a0 = M_bar @ u                                          # [n]
        beta = (M_bar.T @ a0) / a0.pow(2).sum()                 # [j]
        beta = beta.clamp_min(0)
        colsum = M_bar.sum(0)
        f = (beta * a0.sum() / colsum.clamp_min(1e-12)).clamp(0, f_cap)
        back_share = float(a0.sum() * beta.sum() / M_bar.sum())
    print(f"pinned backbone: mean f {f.mean():.3f}  "
          f"mass share {back_share:.3f}", flush=True)

    gen = torch.Generator().manual_seed(seed)
    Wu = (torch.rand(n, k_factors, generator=gen) * 0.5 + 0.2
          ).to(dev).requires_grad_()
    Ws = (torch.rand(k_factors, c_groups, generator=gen) * 0.5 + 0.2
          ).to(dev).requires_grad_()
    Wv = (torch.randn(j, c_groups, generator=gen) * 0.05
          ).to(dev).requires_grad_()
    Wa = torch.log(torch.expm1(a0.clamp_min(1e-6))).clone().requires_grad_()
    opt = torch.optim.Adam([Wu, Ws, Wv, Wa], lr=lr)
    mass = M_bar.sum().clamp_min(1e-8)
    keep1f = (1.0 - f)[:, None]
    chunk = row_chunk if 0 < row_chunk < n else n
    hist = []

    def pieces():
        U = torch.softmax(Wu, dim=1)
        V = keep1f * torch.softmax(Wv, dim=1)
        return U, F.softplus(Ws), V, F.softplus(Wa)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        U, S, V, a = pieces()
        SV = S @ V.T
        loss_val = 0.0
        for i0 in range(0, n, chunk):
            Mh = (U[i0:i0 + chunk] @ SV
                  + a[i0:i0 + chunk, None] * beta[None, :])
            Mb = M_bar[i0:i0 + chunk]
            loss_c = (Mb * ((Mb + 1e-8).log() - (Mh + 1e-8).log())
                      - Mb + Mh).sum() / mass
            loss_c.backward(retain_graph=(i0 + chunk < n))
            loss_val += loss_c.item()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                U, S, V, a = pieces()
                SV = S @ V.T
                resid = sum(float((M_bar[i0:i0 + chunk]
                                   - U[i0:i0 + chunk] @ SV
                                   - a[i0:i0 + chunk, None] * beta[None, :])
                                  .pow(2).sum()) for i0 in range(0, n, chunk))
                rel = (resid ** 0.5) / float(M_bar.norm())
            hist.append(f"step {step} idiv {loss_val:.4e} rel {rel:.4f}")
            print("  " + hist[-1], flush=True)
    with torch.no_grad():
        U, S, V, a = pieces()
    met = {"idiv": loss_val, "rel_err": rel, "back_share": back_share,
           "mean_f": float(f.mean()), "f_cap": f_cap,
           "mean_residual_membership": float(f.mean()), "log": hist[-3:]}
    return U.detach(), S.detach(), V.detach(), f, a.detach(), beta, met


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_events", type=int, default=16_384)
    ap.add_argument("--basis", choices=("svd", "gradpca", "scalar"),
                    default="svd",
                    help="variant B (SVD atoms), variant C (gradient-PCA "
                         "usage-adapted atoms + residual), or variant A "
                         "(scalar atoms: every weight its own atom)")
    ap.add_argument("--r_fc1", type=int, default=100)
    ap.add_argument("--r_fc2", type=int, default=32)
    ap.add_argument("--pca_events", type=int, default=8192)
    ap.add_argument("--k_factors", type=int, default=600)
    ap.add_argument("--c_groups", type=int, default=300)
    ap.add_argument("--fact_steps", type=int, default=4000)
    ap.add_argument("--row_chunk", type=int, default=-1,
                    help="event rows per idiv chunk in the fit; -1 = auto "
                         "(2048 for scalar atoms, full batch otherwise)")
    ap.add_argument("--centering", choices=("none", "lift"), default="none",
                    help="'lift' divides M_bar by its per-atom mean profile "
                         "(ratio to the independence model, the multiplicative "
                         "analog of geo's double-centering) and renormalizes "
                         "rows before fitting")
    ap.add_argument("--pinned", action="store_true",
                    help="pin a rank-1 LS backbone (cofac67 fit_pinned) and "
                         "factorize only each atom's remaining (1-f)")
    ap.add_argument("--f_cap", type=float, default=0.9)
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
    elif args.basis == "scalar":
        A, atom_matrix, sigma = scalar_attributions(model, sig, dev)
    else:
        r_per = {"fc1.weight": args.r_fc1, "fc2.weight": args.r_fc2}
        A, atom_matrix, sigma = gradpca_attributions(model, sig, r_per,
                                                     args.pca_events, dev)
    big = A.shape[1] > 20_000
    if big and dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"A: {tuple(A.shape)} (events x atoms, basis={args.basis}), "
          f"atoms per matrix: fc1={atom_matrix.count('fc1.weight')} "
          f"fc2={atom_matrix.count('fc2.weight')}", flush=True)

    # per-matrix RMS normalization (atoms are block-ordered -> slices),
    # in place; then row-normalized magnitudes
    A_raw = None if big else A.clone()
    j0 = 0
    for name in MATRICES:
        n_m = atom_matrix.count(name)
        sl = slice(j0, j0 + n_m)
        j0 += n_m
        rms = A[:, sl].float().pow(2).mean().sqrt().clamp_min(1e-12)
        A[:, sl] = (A[:, sl].float() / rms).to(A.dtype)
    M_bar = A.abs().float().to(dev)
    M_bar /= M_bar.sum(dim=1, keepdim=True).clamp_min(1e-8)
    if args.centering == "lift":
        col = M_bar.mean(0).clamp_min(1e-10)
        M_bar /= col[None, :]
        M_bar /= M_bar.sum(dim=1, keepdim=True).clamp_min(1e-8)
        print("lift centering: divided by per-atom mean profile, "
              "rows renormalized", flush=True)

    torch.manual_seed(args.seed)
    rc = args.row_chunk if args.row_chunk >= 0 else (2048 if big else 0)
    if args.pinned:
        Uf, Sf, Vf, f, a_bb, beta, met = fit_pinned(
            M_bar, args.k_factors, args.c_groups, steps=args.fact_steps,
            seed=args.seed, f_cap=args.f_cap, row_chunk=rc)
        fact = None
        del M_bar
        V, r = Vf.cpu(), f.cpu()
    else:
        fact = TriFactorizationV2(args.n_events, A.shape[1], args.k_factors,
                                  args.c_groups,
                                  u_simplex=not args.no_u_simplex,
                                  seed=args.seed).to(dev)
        met = fact.fit(M_bar, steps=args.fact_steps, row_chunk=rc)
        del M_bar

    with torch.no_grad():
        if fact is not None:
            V = fact.V.detach().cpu()
            r = fact.residual.detach().cpu()
        Vd = V.to(dev)
        z_rows = []
        for i0 in range(0, A.shape[0], 2048):
            z_rows.append((A[i0:i0 + 2048].to(dev).float() @ Vd).cpu())
        z = torch.cat(z_rows)                         # signed usage [N, C]
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

    if fact is not None:
        Usave, Ssave = fact.U.detach().cpu(), fact.S.detach().cpu()
    else:
        Usave, Ssave = Uf.cpu(), Sf.cpu()
    blob = {"U": Usave, "S": Ssave,
            "V": V, "r": r, "sigma": sigma, "tasks": tasks, "z": z.half(),
            "task_usage": task_usage, "config": met["config"],
            "atom_counts": {n: atom_matrix.count(n) for n in MATRICES}}
    if args.pinned:
        blob["backbone_a"] = a_bb.cpu()
        blob["backbone_beta"] = beta.cpu()
    if not big:                       # raw A too large for scalar atoms
        blob["A"] = A_raw.half()
        blob["atom_matrix"] = atom_matrix
    torch.save(blob, args.out / "factorization.pt")
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
