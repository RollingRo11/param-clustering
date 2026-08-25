"""v2 co-factorization per the blue additions in the updated proposal (Aug 20
PDF): residual/background component + I-divergence objective.

Changes vs run_parfact.py (v1), sec 4.4/4.5 blue text:
  - V rows live on a (C+1)-way softmax: sum_c V_jc <= 1, with the leftover
    r_j = 1 - sum_c V_jc defining an always-retained background component
    C0 = sum_j r_j b_j, so C0 + sum_c C_c = theta exactly. Atoms with no
    reliable attribution structure are no longer force-assigned.
  - primary objective is generalized KL/I-divergence
    D_I(M||M^) = sum M log(M/M^) - M + M^  (--objective euclid for the
    Euclidean ablation).

Fits from the SAME saved M_bar as the v1 run, so the factorization is the
only thing that changes.

    python run_parfact_v2.py --v1 out/B_layer_K1200_C600 --ckpt ...
"""
import argparse
import json
from pathlib import Path

import torch

from induction_model import InductionModel
from atoms import AtomBasis


def fit_v2(M_bar, k, c, steps=4000, lr=2e-2, objective="idiv", seed=0,
           u_simplex=False, s_simplex=False, lambda_v=0.0, eps=1e-8,
           log=print):
    # init + loss normalization mirror cofact.TriFactorization exactly; the
    # only structural change is V's extra residual column (softmax over C+1).
    n, j = M_bar.shape
    g = torch.Generator().manual_seed(seed)
    dev = M_bar.device
    Wu = (torch.rand(n, k, generator=g) * 0.5 + 0.2).to(dev).requires_grad_()
    Ws = (torch.rand(k, c + 1, generator=g) * 0.5 + 0.2 if s_simplex else
          torch.rand(k, c, generator=g) * 0.5 + 0.2).to(dev).requires_grad_()
    Wv = (torch.randn(j, c + 1, generator=g) * 0.05).to(dev).requires_grad_()
    # v1's _rescale_init: scale S so the initial recon matches M_bar in
    # least-squares sense -- without this the fit starts ~1e4x too large
    # and (empirically) never recovers.
    if not s_simplex:      # alpha-rescale only applies to softplus S
        with torch.no_grad():
            U0 = torch.nn.functional.softplus(Wu)
            S0 = torch.nn.functional.softplus(Ws)
            V0 = torch.softmax(Wv, dim=1)[:, :c]
            R = U0 @ S0 @ V0.T
            alpha = (M_bar * R).sum() / R.pow(2).sum().clamp_min(1e-12)
            s_scaled = S0 * alpha.clamp_min(1e-6)
            Ws.copy_(s_scaled + torch.log(-torch.expm1(-s_scaled)))
    opt = torch.optim.Adam([Wu, Ws, Wv], lr=lr)
    denom = M_bar.pow(2).sum().clamp_min(eps)
    mass = M_bar.sum().clamp_min(eps)
    for step in range(steps):
        U = (torch.softmax(Wu, dim=1) if u_simplex
             else torch.nn.functional.softplus(Wu))
        S = (torch.softmax(Ws, dim=1)[:, :c] if s_simplex
             else torch.nn.functional.softplus(Ws))
        Vfull = torch.softmax(Wv, dim=1)
        V = Vfull[:, :c]
        M_hat = U @ S @ V.T
        if objective == "idiv":
            loss = (M_bar * ((M_bar + eps).log() - (M_hat + eps).log())
                    - M_bar + M_hat).sum() / mass
        else:
            loss = (M_bar - M_hat).pow(2).sum() / denom
        if lambda_v:
            # row entropy over ALL C+1 slots: atoms may collapse into one
            # component OR the residual/background -- pruning pressure
            loss = loss + lambda_v * -(Vfull * (Vfull + 1e-12).log())                 .sum(1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            with torch.no_grad():
                rel = ((M_bar - M_hat).norm() / M_bar.norm()).item()
            log(f"v2 step {step:5d}  {objective} {loss.item():.4e}  "
                f"rel_err {rel:.5f}")
    with torch.no_grad():
        U = (torch.softmax(Wu, dim=1) if u_simplex
             else torch.nn.functional.softplus(Wu))
        S = (torch.softmax(Ws, dim=1)[:, :c] if s_simplex
             else torch.nn.functional.softplus(Ws))
        Vfull = torch.softmax(Wv, dim=1)
    return U.detach(), S.detach(), Vfull[:, :c].detach(), \
        Vfull[:, c].detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, required=True,
                    help="v1 run dir with factorization.pt (source of M_bar)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--objective", choices=["idiv", "euclid"], default="idiv")
    ap.add_argument("--lambda_v", type=float, default=0.0,
                    help="entropy penalty on V rows (incl. residual slot)")
    ap.add_argument("--s_simplex", action="store_true",
                    help="S rows on a <=1 simplex (softmax over C+1)")
    ap.add_argument("--u_simplex", action="store_true",
                    help="normalize U rows onto the simplex (matches the v1 "
                         "flag; off by default for parity with the v1 run)")
    ap.add_argument("--fact_steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    out = args.out or Path(str(args.v1) + f"_v2_{args.objective}"
                           + ("_usimplex" if args.u_simplex else "")
                           + ("_ssimplex" if args.s_simplex else "")
                           + (f"_lv{args.lambda_v:g}" if args.lambda_v
                              else ""))
    out.mkdir(parents=True, exist_ok=True)

    fact = torch.load(args.v1 / "factorization.pt", weights_only=False,
                      map_location=dev)
    M_bar = fact["M_bar"].to(dev).float()
    A_signed = fact["A_signed"].to(dev).float()
    k = fact["U"].shape[1]
    c = fact["V"].shape[1]
    print(f"M_bar {tuple(M_bar.shape)}  K={k} C={c}  obj={args.objective}")

    U, S, V, r = fit_v2(M_bar, k, c, steps=args.fact_steps,
                        objective=args.objective, seed=args.seed,
                        u_simplex=args.u_simplex,
                        s_simplex=args.s_simplex, lambda_v=args.lambda_v)

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    matrices = sorted(set(fact["atom_matrix"]), key=fact["atom_matrix"].index)
    basis = AtomBasis.build(model, matrices, fact["config"]["variant"])
    comps = basis.components(V)                      # candidates only
    C0 = basis.components(r.unsqueeze(1))            # background, [1, o, i]
    z = A_signed @ V

    # exactness: C0 + sum_c C_c must equal the decomposed weights
    err = 0.0
    for nm in comps:
        W = dict(model.named_parameters())[nm].detach()
        err = max(err, (comps[nm].sum(0) + C0[nm][0] - W).abs().max().item())

    with torch.no_grad():
        M_hat = U @ S @ V.T
        r2 = 1 - (M_bar - M_hat).pow(2).sum().item() / \
            (M_bar - M_bar.mean()).pow(2).sum().item()
        res_frac_atoms = r.mean().item()
        c0_mass = sum(C0[nm][0].square().sum().item() for nm in C0)
        th_mass = sum(dict(model.named_parameters())[nm].square().sum().item()
                      for nm in C0)

    metrics = {"objective": args.objective, "r2_attr_euclid": r2,
               "recon_max_err_with_C0": err,
               "mean_residual_membership": res_frac_atoms,
               "C0_mass_frac": c0_mass / th_mass}
    print(json.dumps(metrics, indent=1))

    torch.save({"U": U.cpu(), "S": S.cpu(), "V": V.cpu(), "r": r.cpu(),
                "A_signed": A_signed.cpu(), "M_bar": M_bar.cpu(),
                "z": z.cpu(),
                "components": {n: t.cpu() for n, t in comps.items()},
                "C0": {n: t[0].cpu() for n, t in C0.items()},
                "atom_matrix": fact["atom_matrix"],
                "config": {**fact["config"], "v2_objective": args.objective}},
               out / "factorization.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
