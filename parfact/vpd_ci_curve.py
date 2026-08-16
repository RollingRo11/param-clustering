"""Per-token VPD curve ordered by VPD's own causal-importance function.

For each event, each of the 100 clustered components is scored by its member
subcomponents' causal importances from the trained CI transformer
(per-subcomponent max over sequence positions, summed over cluster members),
and ablated cumulatively least-important-first — VPD's native cheap ordering,
the analog of z_ic for the co-factorization. Appends the curve to the run's
ablation_curve.json as spec 'per_example_asc:ci'.

    python vpd_ci_curve.py --run out/vpd_C100
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

from induction_model import InductionModel
from atoms import make_events
from ablation_curve import build_ranks
from vpd_toy import MODULES, CITransformer, install_components, clear_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=Path(__file__).parent / "out/vpd_C100")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    dev = args.device

    blob = torch.load(args.run / "components.pt", weights_only=True,
                      map_location=dev)
    comps = {n: t.float() for n, t in blob["components"].items()}
    lab = blob["labels"].to(dev)
    n_comp = next(iter(comps.values())).shape[0]
    state = torch.load(args.run / "vpd_state.pt", weights_only=True,
                       map_location=dev)

    # clean model for the curve itself
    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # wrapped copy of the model to run the CI function
    wrapped = InductionModel().to(dev)
    wrapped.load_state_dict(torch.load(args.ckpt)["state_dict"])
    wrapped.eval()
    wrappers = install_components(wrapped, state["ci_fn"]["proj_out.weight"]
                                  .shape[0] // len(MODULES))
    wrapped.to(dev)
    c_per = wrappers[MODULES[0]].C
    for n, w in wrappers.items():
        w.V.data.copy_(state["wrappers"][n]["V"].to(dev))
        w.U.data.copy_(state["wrappers"][n]["U"].to(dev))
    ci_fn = CITransformer({n: 16 for n in MODULES}, c_per).to(dev)
    ci_fn.load_state_dict(state["ci_fn"])
    ci_fn.eval()

    events = make_events(model, args.n_seq, "final", seed=args.seed + 1000)
    seq, m_tok = events["seq"], events["m_token"]
    n_ev = seq.shape[0]

    # -- per-event component importance from causal importances --------------
    with torch.no_grad():
        scores = []
        for i in range(0, n_ev, args.chunk):
            clear_masks(wrappers)
            wrapped(seq[i: i + args.chunk])
            ci_lower, _ = ci_fn({n: w.last_input
                                 for n, w in wrappers.items()})
            # [B, 8*C]: per-subcomponent max CI over positions, module order
            # matching the clustering (sorted paths)
            scores.append(torch.cat([ci_lower[n].amax(dim=1)
                                     for n in sorted(MODULES)], dim=1))
        sub_score = torch.cat(scores)                  # [N, 800]
        imp = torch.zeros(n_ev, n_comp, device=dev)
        imp.index_add_(1, lab, sub_score)
    print(f"CI importance [{n_ev}, {n_comp}] "
          f"(mean active comps/event at >0.1: "
          f"{(imp > 0.1).float().sum(1).mean():.1f})")

    # -- cumulative per-token ablation curve ---------------------------------
    params = {n: p.detach() for n, p in model.named_parameters()}
    in_dims = ({n: (0 if n in comps else None) for n in params}, 0)

    def fwd(pdict, s):
        return functional_call(model, pdict, (s.unsqueeze(0),))[0, -1]

    batched_fwd = vmap(fwd, in_dims=in_dims)
    rank = build_ranks("per_example_asc", imp, args.seed)
    curve = []
    with torch.no_grad():
        for k in range(n_comp + 1):
            ces, hits = [], []
            for i in range(0, n_ev, args.chunk):
                sl = slice(i, i + args.chunk)
                mask = (rank[sl] < k).float()
                pdict = dict(params)
                for name, ct in comps.items():
                    pdict[name] = params[name] - (mask @ ct.reshape(
                        n_comp, -1)).reshape(-1, *ct.shape[1:])
                logits = batched_fwd(pdict, seq[sl])
                ces.append(F.cross_entropy(logits, m_tok[sl],
                                           reduction="none"))
                hits.append(logits.argmax(-1) == m_tok[sl])
            ce = torch.cat(ces).mean().item()
            acc = torch.cat(hits).float().mean().item()
            curve.append({"k": k, "ce": round(ce, 6), "acc": round(acc, 5)})
            if k % 20 == 0 or k == n_comp:
                print(f"  K={k:<4} CE {ce:9.4f}  acc {acc:.4f}", flush=True)
    base_ce = curve[0]["ce"]
    for r in curve:
        r["delta"] = round(r["ce"] - base_ce, 6)

    out = json.load(open(args.run / "ablation_curve.json"))
    out["curves"]["per_example_asc:ci"] = curve

    def acc_budget(c, floor):
        ok = [r["k"] for r in c if all(r2["acc"] >= floor
                                       for r2 in c if r2["k"] <= r["k"])]
        return max(ok) if ok else 0

    out["components_removable_keeping_acc"]["per_example_asc:ci"] = {
        f"{fl}": acc_budget(curve, fl) for fl in (0.99, 0.95, 0.9)}
    (args.run / "ablation_curve.json").write_text(json.dumps(out, indent=1))
    print("removable keeping acc:",
          out["components_removable_keeping_acc"]["per_example_asc:ci"])


if __name__ == "__main__":
    main()
