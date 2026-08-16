"""End-to-end attribution-based co-factorization on the toy induction model.

Implements section 4 of the proposal: prediction events -> parameter-atom
attributions -> normalized allocation matrix -> nonnegative tri-factorization
M_bar ~= U S V^T -> additive parameter components C_c = sum_j V_jc b_j and
cheap per-event component usage z_ic = sum_j V_jc a_ij.

Examples:
  python run_parfact.py --train --train_steps 100000       # once, saves ckpt
  python run_parfact.py --variant B --norm layer
  python run_parfact.py --variant C --norm fisher --k_factors 8 --c_groups 6
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from induction_model import InductionModel, train_induction, eval_induction
from atoms import (ATTN_MATRICES, EMBED_MATRICES, AtomBasis, collect_grads,
                   estimate_fisher, make_events)
from cofact import (TriFactorization, allocation_matrix, component_usage,
                    effective_number, group_mass, normalize_attributions)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 0 else float("nan")


@torch.no_grad()
def ablation_check(model, basis, comps, events, z, n_events: int = 512):
    """Sec 5 smoke test: actual ablation effect Delta_ic = s_i(theta) -
    s_i(theta - C_c) vs the first-order prediction z_ic, per component."""
    device = z.device
    idx = torch.randperm(z.shape[0], device=device)[:n_events]
    seq, pos, y = events["seq"][idx], events["pos"][idx], events["y"][idx]
    rows = torch.arange(idx.shape[0], device=device)

    def score(m):
        return F.log_softmax(m(seq)[rows, pos], dim=-1)[rows, y]

    base = score(model)
    n_comp = z.shape[1]
    out = []
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    for c in range(n_comp):
        for name in basis.matrices:
            params[name].sub_(comps[name][c])
        for name, p in model.named_parameters():
            p.copy_(params[name])
        delta = base - score(model)
        for name in basis.matrices:            # restore
            params[name].add_(comps[name][c])
        out.append({"component": c,
                    "pearson_r": pearson(z[idx, c], delta),
                    "mean_abs_delta": delta.abs().mean().item()})
    for name, p in model.named_parameters():   # restore model exactly
        p.copy_(params[name])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B", "C"], default="B",
                    help="A: scalar atoms; B: SVD atoms (sigma-weighted); "
                         "C: unit outer-product singular-vector directions")
    ap.add_argument("--norm", choices=["layer", "fisher", "none"],
                    default="layer")
    ap.add_argument("--norm_group", choices=["layer", "matrix"],
                    default="layer",
                    help="grouping for --norm layer: per depth group (paper) "
                         "or per source weight matrix")
    ap.add_argument("--include_embed", action="store_true",
                    help="also decompose embed/pos/unembed matrices")
    ap.add_argument("--n_seq", type=int, default=4096)
    ap.add_argument("--positions", choices=["final", "mixed", "all"],
                    default="mixed",
                    help="prediction events per sequence: induction position "
                         "only, plus random earlier positions, or every pos")
    ap.add_argument("--k_factors", type=int, default=8,
                    help="K behavioral factors (rows of S)")
    ap.add_argument("--c_groups", type=int, default=6,
                    help="C parameter groups / components (cols of S)")
    ap.add_argument("--u_simplex", action="store_true",
                    help="normalize U rows onto the simplex")
    ap.add_argument("--lambda_u", type=float, default=0.0)
    ap.add_argument("--lambda_v", type=float, default=0.0)
    ap.add_argument("--fact_steps", type=int, default=4000)
    ap.add_argument("--fact_lr", type=float, default=2e-2)
    ap.add_argument("--fisher_samples", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true",
                    help="train the toy model even if a checkpoint exists")
    ap.add_argument("--train_steps", type=int, default=100_000)
    ap.add_argument("--no_ablation_check", action="store_true")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir; default out/<variant>_<norm>[_embed]")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    tag = f"{args.variant}_{args.norm}" + ("_embed" if args.include_embed else "")
    out_dir = args.out or Path(__file__).parent / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # -- target model -------------------------------------------------------
    model = InductionModel().to(args.device)
    if args.train or not args.ckpt.exists():
        print("training toy induction model...")
        acc = train_induction(model, steps=args.train_steps, seed=args.seed)
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "accuracy": acc},
                   args.ckpt)
    else:
        model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    acc = eval_induction(model)
    print(f"induction accuracy: {acc:.4f}")

    # -- step 0-2: events, atoms, attribution matrix ------------------------
    matrices = list(ATTN_MATRICES) + (list(EMBED_MATRICES)
                                      if args.include_embed else [])
    basis = AtomBasis.build(model, matrices, args.variant)
    print(f"variant {args.variant}: {basis.n_atoms} atoms over "
          f"{len(matrices)} matrices")

    events = make_events(model, args.n_seq, args.positions, seed=args.seed)
    n_events = events["seq"].shape[0]
    print(f"{n_events} prediction events ({args.positions})")

    grads = collect_grads(model, matrices, events["seq"], events["pos"],
                          events["y"])
    A_signed = basis.attributions(grads)
    del grads

    fisher = None
    if args.norm == "fisher":
        print("estimating diagonal Fisher in atom space...")
        f_seq = events["seq"][: min(2048, n_events)]
        f_pos = events["pos"][: min(2048, n_events)]
        fisher = estimate_fisher(model, basis, f_seq, f_pos,
                                 n_samples=args.fisher_samples, seed=args.seed)
    groups = (basis.atom_layer if args.norm_group == "layer"
              else basis.atom_matrix)
    A_tilde = normalize_attributions(A_signed, args.norm, groups=groups,
                                     fisher=fisher, eps=args.eps)
    M_bar = allocation_matrix(A_tilde, eps=args.eps)

    # -- step 3: joint soft co-factorization --------------------------------
    fact = TriFactorization(n_events, basis.n_atoms, args.k_factors,
                            args.c_groups, u_simplex=args.u_simplex,
                            seed=args.seed).to(args.device)
    stats = fact.fit(M_bar, steps=args.fact_steps, lr=args.fact_lr,
                     lambda_u=args.lambda_u, lambda_v=args.lambda_v)
    U, S, V = fact.U.detach(), fact.S.detach(), fact.V.detach()
    print(f"R^2_attr = {stats['r2_attr']:.4f}")

    # -- step 4-5: components + cheap usage ---------------------------------
    comps = basis.components(V)
    recon_err = max((comps[n].sum(0) - basis.weights[n]).abs().max().item()
                    for n in matrices)
    print(f"sum_c C_c == theta check: max abs err {recon_err:.2e}")
    z = component_usage(A_signed, V)

    # -- diagnostics --------------------------------------------------------
    metrics = {
        "accuracy": acc, "n_events": n_events, "n_atoms": basis.n_atoms,
        **stats, "component_recon_max_err": recon_err,
        "u_effective_factors_mean": effective_number(U).mean().item(),
        "v_effective_groups_mean": effective_number(V).mean().item(),
        "group_mass_matrix": group_mass(V, basis.atom_matrix),
        "group_mass_layer": group_mass(V, basis.atom_layer),
        # does statistical usage (US)_ic track first-order usage |z_ic|?
        "usage_corr_per_component": [
            pearson((U @ S)[:, c], z[:, c].abs()) for c in range(args.c_groups)],
    }
    # relative component usage on induction events vs other events (raw |z| is
    # dominated by gradient scale: the model is near-saturated at the
    # induction position, so each event's usage is normalized to shares)
    z_share = z.abs() / z.abs().sum(1, keepdim=True).clamp_min(1e-12)
    ind = events["is_induction"]
    metrics["induction_usage_share_ratio"] = [
        (z_share[ind, c].mean() / z_share[~ind, c].mean().clamp_min(1e-12)).item()
        if (~ind).any() else float("nan") for c in range(args.c_groups)]
    if not args.no_ablation_check:
        metrics["ablation_check"] = ablation_check(model, basis, comps,
                                                   events, z)

    torch.save({"U": U.cpu(), "S": S.cpu(), "V": V.cpu(),
                "A_signed": A_signed.cpu(), "M_bar": M_bar.cpu(),
                "z": z.cpu(),
                "components": {n: c.cpu() for n, c in comps.items()},
                "atom_matrix": basis.atom_matrix,
                "atom_layer": basis.atom_layer,
                "events": {k: v.cpu() for k, v in events.items()},
                "config": vars(args) | {"out": str(out_dir),
                                        "ckpt": str(args.ckpt)}},
               out_dir / "factorization.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("group_mass_matrix",)}, indent=1))
    print(f"saved {out_dir}/factorization.pt and metrics.json")


if __name__ == "__main__":
    main()
