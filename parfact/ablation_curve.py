"""Minimality curve: induction CE as components are ablated in importance order.

The analogue of geo-attribution/ablation_curve.py for the co-factorization
decomposition. Components are additive (sum_c C_c = theta), so K=0 reproduces
the target bit-for-bit and there is no reconstruction error in the curve —
what is measured is pure minimality: how many components can be deleted before
the induction prediction goes.

Unlike the previous method's global orderings, z_ic gives a cheap PER-EXAMPLE
importance, so the headline ordering here ablates, for every event separately,
that event's least-important components first (per_example_asc). Global
(mean-|z|) and random orderings are the baselines; global_desc is the
contrast case (delete the most important first).

Right endpoint: ablating all C components zeroes the decomposed attention
matrices only — the residual embed -> unembed path survives, so the end state
is a bigram-ish model, not a uniform one. ln(128) = 4.85 nats is the uniform
reference; anything above it is confidently wrong rather than uninformed.

    python ablation_curve.py --run out/B_layer_K400_C100_long
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

from induction_model import VOCAB, InductionModel
from atoms import AtomBasis, collect_attributions, make_events

ORDER_LABELS = {
    "per_example_asc": "per-example, least important first",
    "global_asc": "global, least important first",
    "random": "random order",
    "global_desc": "global, most important first",
}


def build_ranks(order: str, imp: torch.Tensor, seed: int) -> torch.Tensor:
    """rank[n, c] = position of component c in event n's ablation order."""
    n_events, n_comp = imp.shape
    if order == "per_example_asc":
        return imp.argsort(dim=1).argsort(dim=1)
    if order == "global_asc":
        srt = imp.mean(0).argsort()
    elif order == "global_desc":
        srt = imp.mean(0).argsort(descending=True)
    elif order == "random":
        g = torch.Generator(device=imp.device).manual_seed(seed)
        srt = torch.randperm(n_comp, device=imp.device, generator=g)
    else:
        raise ValueError(order)
    rank = torch.empty(n_comp, dtype=torch.long, device=imp.device)
    rank[srt] = torch.arange(n_comp, device=imp.device)
    return rank.expand(n_events, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=Path(__file__).parent / "out/B_layer_K400_C100_long")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--orders", nargs="+", default=list(ORDER_LABELS))
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    dev = args.device

    fact = torch.load(args.run / "factorization.pt", weights_only=False,
                      map_location=dev)
    cfg = fact["config"]
    V = fact["V"].to(dev)
    n_comp = V.shape[1]

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    matrices = sorted(set(fact["atom_matrix"]),
                      key=fact["atom_matrix"].index)
    basis = AtomBasis.build(model, matrices, cfg["variant"])
    comps = basis.components(V)                       # name -> [C, out, in]

    # fresh induction events; importance of component c to the token = |z_ic|
    events = make_events(model, args.n_seq, "final", seed=args.seed + 1000)
    seq, pos, y = events["seq"], events["pos"], events["y"]
    m_tok = events["m_token"]
    z = collect_attributions(model, basis, seq, pos, y) @ V
    imp = z.abs()

    params = {n: p.detach() for n, p in model.named_parameters()}
    in_dims = ({n: (0 if n in comps else None) for n in params}, 0)

    def fwd(pdict, s):
        return functional_call(model, pdict, (s.unsqueeze(0),))[0, -1]

    batched_fwd = vmap(fwd, in_dims=in_dims)

    @torch.no_grad()
    def ce_at(rank: torch.Tensor, k: int) -> tuple[float, float]:
        """Mean CE (nats) of the true m-token and accuracy, each event with its
        own first-k-of-its-order components removed."""
        ces, hits = [], []
        for i in range(0, seq.shape[0], args.chunk):
            sl = slice(i, i + args.chunk)
            mask = (rank[sl] < k).to(V.dtype)          # [B, C]
            pdict = dict(params)
            for name, ct in comps.items():
                delta = (mask @ ct.reshape(n_comp, -1)).reshape(
                    -1, *ct.shape[1:])
                pdict[name] = params[name] - delta
            logits = batched_fwd(pdict, seq[sl])
            ces.append(F.cross_entropy(logits, m_tok[sl], reduction="none"))
            hits.append(logits.argmax(-1) == m_tok[sl])
        return (torch.cat(ces).mean().item(),
                torch.cat(hits).float().mean().item())

    ks = list(range(n_comp + 1))
    base_ce, base_acc = ce_at(torch.zeros_like(imp, dtype=torch.long), 0)
    print(f"base CE {base_ce:.5f}  acc {base_acc:.4f}  "
          f"uniform {math.log(VOCAB):.3f}  C={n_comp}")

    curves = {}
    for order in args.orders:
        rank = build_ranks(order, imp, args.seed)
        curve = []
        for k in ks:
            ce, acc = ce_at(rank, k)
            curve.append({"k": k, "ce": round(ce, 5), "acc": round(acc, 5),
                          "delta": round(ce - base_ce, 5)})
            if k % 10 == 0 or k == n_comp:
                print(f"  {order:<16} K={k:<4} CE {ce:9.4f}  acc {acc:.4f}",
                      flush=True)
        curves[order] = curve

    def budget(curve, tol):
        ok = [r["k"] for r in curve if r["delta"] <= tol]
        return max(ok) if ok else 0

    out = {"format": "parfact_ablation_curve_v1", "C": n_comp,
           "run": str(args.run), "n_events": int(seq.shape[0]),
           "base_ce": round(base_ce, 5), "uniform_ce": round(math.log(VOCAB), 5),
           "curves": curves,
           "components_removable_within": {
               name: {f"{t}": budget(c, t) for t in (0.01, 0.05, 0.1, 0.5)}
               for name, c in curves.items()}}
    (args.run / "ablation_curve.json").write_text(json.dumps(out, indent=1))
    for name, b in out["components_removable_within"].items():
        print(f"{name:<16} removable within ΔCE: " +
              "  ".join(f"{t}: {v} ({100 * v / n_comp:.0f}%)"
                        for t, v in b.items()))
    plot(out, args.run / "ablation_curve.png")
    print(f"wrote {args.run}/ablation_curve.json and ablation_curve.png")


def plot(out: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # categorical slots 1-4 (validated adjacent order), light mode
    colors = {"per_example_asc": "#2a78d6", "global_asc": "#eb6834",
              "random": "#1baf7a", "global_desc": "#eda100"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for order, curve in out["curves"].items():
        ks = [r["k"] for r in curve]
        ce = [r["ce"] for r in curve]
        ax.plot(ks, ce, lw=2, color=colors.get(order, "#898781"),
                label=ORDER_LABELS.get(order, order))
    ax.axhline(out["uniform_ce"], color="#898781", lw=1, ls=(0, (4, 3)))
    ax.text(out["C"], out["uniform_ce"] * 0.78, "uniform ln(128)",
            fontsize=8, color="#898781", ha="right")
    # symlog: the near-zero plateau (the whole point of the asc orderings)
    # stays visible instead of falling off a log axis
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=-1e-3)
    ax.set_xlabel(f"components ablated (of {out['C']}, in each ordering)",
                  color="#52514e")
    ax.set_ylabel("induction CE on true m-token (nats, symlog)",
                  color="#52514e")
    ax.set_title("Ablating co-factorization components by importance order",
                 color="#0b0b0b", fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", lw=0.75)
    ax.tick_params(colors="#898781")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e",
              loc="lower left", bbox_to_anchor=(0.14, 0.10))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    main()
