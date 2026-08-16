"""Per-token minimality: does keeping only the components that matter for a
token still generate that token?

For every prediction event separately, components are ablated in that event's
own least-important-first order (sec 4.6: z_ic = grad s_i . C_c is a
per-event usage profile), and we measure CE and generation accuracy of the
event's m-token. Orderings are given as `name[:score]`:

  per_example_asc:logp     |z| with s = log p(y|x)  — the paper's default;
                           weak at saturated positions, where grad log p -> 0
  per_example_asc:logodds  |z| with s = logit_y - logsumexp(other logits) —
                           same single backward, non-saturating
  per_example_asc:oracle   true single-component ablation |delta log p| per
                           event: the upper bound any per-token score can hit
  global_asc / random / global_desc: baselines (global uses mean-|z|)

K=0 reproduces the target bit-for-bit (components are exactly additive), so
the curve is pure minimality. Right endpoint: all C ablated zeroes the
decomposed attention matrices but keeps the embed->unembed path, a bigram-ish
model, not a uniform one (ln 128 = 4.85 nats).

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
    "per_example_asc": "per-token, least important first",
    "global_asc": "global, least important first",
    "random": "random order",
    "global_desc": "global, most important first",
}
SCORE_LABELS = {"logp": "z from log p", "logit": "z from logit",
                "logodds": "z from log-odds",
                "oracle": "true single-ablation Δ"}


def spec_label(spec: str) -> str:
    order, _, score = spec.partition(":")
    lab = ORDER_LABELS.get(order, order)
    return f"{lab} ({SCORE_LABELS[score]})" if score else lab


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
    ap.add_argument("--orders", nargs="+",
                    default=["per_example_asc:logp", "per_example_asc:logodds",
                             "per_example_asc:oracle", "global_asc:logp"])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--components", type=Path, default=None,
                    help="load components from this .pt (dict name -> "
                         "[C, out, in]) instead of a factorization run; only "
                         "oracle/random orderings are available then")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    dev = args.device

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if args.components is not None:
        blob = torch.load(args.components, weights_only=True, map_location=dev)
        comps = {n: t.float() for n, t in blob["components"].items()}
        n_comp = next(iter(comps.values())).shape[0]
        basis = None
        run_dir = args.components.parent
    else:
        fact = torch.load(args.run / "factorization.pt", weights_only=False,
                          map_location=dev)
        cfg = fact["config"]
        V = fact["V"].to(dev)
        n_comp = V.shape[1]
        matrices = sorted(set(fact["atom_matrix"]),
                          key=fact["atom_matrix"].index)
        basis = AtomBasis.build(model, matrices, cfg["variant"])
        comps = basis.components(V)                   # name -> [C, out, in]
    run_dir = args.components.parent if args.components else args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    events = make_events(model, args.n_seq, "final", seed=args.seed + 1000)
    seq, pos, y = events["seq"], events["pos"], events["y"]
    m_tok = events["m_token"]

    params = {n: p.detach() for n, p in model.named_parameters()}
    in_dims = ({n: (0 if n in comps else None) for n in params}, 0)

    def fwd(pdict, s):
        return functional_call(model, pdict, (s.unsqueeze(0),))[0, -1]

    batched_fwd = vmap(fwd, in_dims=in_dims)

    @torch.no_grad()
    def oracle_importance() -> torch.Tensor:
        """|delta log p(m-token)| of ablating each component alone, per event."""
        base = F.log_softmax(model(seq)[:, -1], -1).gather(
            1, m_tok[:, None]).squeeze(1)
        imp = torch.zeros(seq.shape[0], n_comp, device=dev)
        for c in range(n_comp):
            pdict = dict(params)
            for name, ct in comps.items():
                pdict[name] = params[name] - ct[c]
            logits = functional_call(model, pdict, (seq,))[:, -1]
            lp = F.log_softmax(logits, -1).gather(1, m_tok[:, None]).squeeze(1)
            imp[:, c] = (base - lp).abs()
        return imp

    imps: dict[str, torch.Tensor] = {}

    def importance(score: str) -> torch.Tensor:
        if score not in imps:
            if score == "oracle":
                imps[score] = oracle_importance()
            else:
                if basis is None:
                    raise SystemExit(f"score '{score}' needs a factorization "
                                     "run; --components only supports oracle")
                z = collect_attributions(model, basis, seq, pos, y,
                                         score=score) @ V
                imps[score] = z.abs()
        return imps[score]

    @torch.no_grad()
    def ce_at(rank: torch.Tensor, k: int) -> tuple[float, float]:
        """Mean CE (nats) of the true m-token and accuracy, each event with its
        own first-k-of-its-order components removed."""
        ces, hits = [], []
        for i in range(0, seq.shape[0], args.chunk):
            sl = slice(i, i + args.chunk)
            mask = (rank[sl] < k).float()          # [B, C]
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
    base_ce, base_acc = ce_at(torch.zeros(seq.shape[0], n_comp,
                                          dtype=torch.long, device=dev), 0)
    print(f"base CE {base_ce:.3e}  acc {base_acc:.4f}  "
          f"uniform {math.log(VOCAB):.3f}  C={n_comp}")

    curves = {}
    for spec in args.orders:
        order, _, score = spec.partition(":")
        rank = build_ranks(order, importance(score or "logp"), args.seed)
        curve = []
        for k in ks:
            ce, acc = ce_at(rank, k)
            curve.append({"k": k, "ce": round(ce, 6), "acc": round(acc, 5),
                          "delta": round(ce - base_ce, 6)})
            if k % 20 == 0 or k == n_comp:
                print(f"  {spec:<26} K={k:<4} CE {ce:9.4f}  acc {acc:.4f}",
                      flush=True)
        curves[spec] = curve

    def acc_budget(curve, floor):
        ok = [r["k"] for r in curve if all(
            r2["acc"] >= floor for r2 in curve if r2["k"] <= r["k"])]
        return max(ok) if ok else 0

    out = {"format": "parfact_ablation_curve_v2", "C": n_comp,
           "run": str(run_dir), "n_events": int(seq.shape[0]),
           "base_ce": round(base_ce, 8), "uniform_ce": round(math.log(VOCAB), 5),
           "curves": curves,
           "components_removable_keeping_acc": {
               spec: {f"{fl}": acc_budget(c, fl) for fl in (0.99, 0.95, 0.9)}
               for spec, c in curves.items()}}
    (run_dir / "ablation_curve.json").write_text(json.dumps(out, indent=1))
    for spec, b in out["components_removable_keeping_acc"].items():
        print(f"{spec:<26} removable keeping acc: " +
              "  ".join(f">={fl}: {v} ({100 * v / n_comp:.0f}%)"
                        for fl, v in b.items()))
    plot_acc(out, run_dir / "ablation_curve_acc.png")
    plot_ce(out, run_dir / "ablation_curve.png")
    print(f"wrote {run_dir}/ablation_curve.json, ablation_curve_acc.png, "
          "ablation_curve.png")


def _style(ax, fig):
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e1e0d9", lw=0.75)
    ax.tick_params(colors="#898781")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")


# categorical slots 1-4 (validated adjacent order), light mode, keyed by spec
SPEC_COLORS = {"per_example_asc:logp": "#2a78d6",
               "per_example_asc:logodds": "#eb6834",
               "per_example_asc:oracle": "#1baf7a",
               "global_asc:logp": "#eda100"}


def plot_acc(out: dict, path: Path, specs: list[str] | None = None):
    """Per-token generation retention: fraction of events whose m-token is
    still argmax after ablating K components in each event's own order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for spec, curve in out["curves"].items():
        if specs is not None and spec not in specs:
            continue
        ax.plot([r["k"] for r in curve], [r["acc"] for r in curve], lw=2,
                color=SPEC_COLORS.get(spec, "#898781"), label=spec_label(spec))
    ax.axhline(1.0, color="#898781", lw=1, ls=(0, (4, 3)))
    ax.text(out["C"], 1.012, "no ablation: 100% of tokens generated",
            fontsize=8, color="#898781", ha="right")
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel(f"components ablated per token (of {out['C']}, "
                  "least important first for that token)", color="#52514e")
    ax.set_ylabel("fraction of tokens still generated", color="#52514e")
    ax.set_title("Keeping only the components that matter for each token",
                 color="#0b0b0b", fontsize=11)
    _style(ax, fig)
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e",
              loc="lower left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def plot_ce(out: dict, path: Path, specs: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for spec, curve in out["curves"].items():
        if specs is not None and spec not in specs:
            continue
        ax.plot([r["k"] for r in curve], [r["ce"] for r in curve], lw=2,
                color=SPEC_COLORS.get(spec, "#898781"), label=spec_label(spec))
    ax.axhline(out["uniform_ce"], color="#898781", lw=1, ls=(0, (4, 3)))
    ax.text(out["C"], out["uniform_ce"] * 0.78, "uniform ln(128)",
            fontsize=8, color="#898781", ha="right")
    ax.axhline(out["base_ce"], color="#898781", lw=1, ls=(0, (4, 3)))
    base_str = f"{out['base_ce']:.2g}" if out["base_ce"] > 0 else "~0"
    ax.annotate(f"model CE, no ablation ({base_str} nats)",
                (0, out["base_ce"]), xytext=(6, 5),
                textcoords="offset points", fontsize=8, color="#898781")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=-1e-3)
    ax.set_xlabel(f"components ablated per token (of {out['C']}, "
                  "least important first for that token)", color="#52514e")
    ax.set_ylabel("induction CE on true m-token (nats, symlog)",
                  color="#52514e")
    ax.set_title("Ablating co-factorization components by importance order",
                 color="#0b0b0b", fontsize=11)
    _style(ax, fig)
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e",
              loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    main()
