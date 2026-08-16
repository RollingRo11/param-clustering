"""Overlay minimality curves: co-factorization vs the previous method.

Both curves ablate components least-important-first and plot the CE increase.
- parfact: global mean-|z_ic| order on the toy induction model (C=100),
  from out/<run>/ablation_curve.json.
- previous method (geo-attribution VPD shares on Llama-3.2-1B, Pile eval,
  C=4096): mass_asc order, parsed from geo-attribution/out/ablation_curve.log
  because the /dev/shm artifacts of that run no longer exist.

Axes are normalized for the cross-setup comparison: x is the FRACTION of
components ablated, y is delta CE in nats (base CEs differ: ~1e-6 vs 0.4529).

    python compare_ablation.py --run out/B_layer_K400_C100_long
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = Path(__file__).parent.parent / "geo-attribution/out/ablation_curve.log"
LINE = re.compile(r"mass_asc\s+K=(\d+)\s+\(\s*[\d.]+%\)\s+CE\s+([\d.]+)\s+"
                  r"Δ\s+([+-][\d.]+)")


def parse_prev(log_path: Path):
    pts = [(int(k), float(ce), float(d))
           for k, ce, d in LINE.findall(log_path.read_text())]
    assert pts, f"no mass_asc points found in {log_path}"
    c = max(k for k, _, _ in pts)
    return {"C": c, "points": [{"k": k, "ce": ce, "delta": d}
                               for k, ce, d in sorted(pts)]}


def plot_oracle_pair(ours: dict, prev: dict, spec: str, path: Path,
                     vpd: dict | None = None):
    """Delta-CE per-token oracle curves for the toy decompositions."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    series = [("co-factorization (this method)", ours, spec, "#2a78d6"),
              ("attribution clustering (previous method)", prev, spec,
               "#eb6834")]
    if vpd is not None:
        series.append(("VPD, single-ablation order", vpd, spec, "#1baf7a"))
        if "per_example_asc:ci" in vpd["curves"]:
            series.append(("VPD, causal-importance order", vpd,
                           "per_example_asc:ci", "#eda100"))
    for label, blob, sp, color in series:
        curve = blob["curves"][sp]
        ax.plot([r["k"] for r in curve], [r["delta"] for r in curve], lw=2,
                color=color, label=label)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.text(1, 0, "no ablation (ΔCE = 0)", fontsize=8, color="#898781",
            va="bottom")
    ax.axhline(ours["uniform_ce"], color="#898781", lw=1, ls=(0, (4, 3)))
    ax.text(ours["C"], ours["uniform_ce"] * 0.72, "uniform ln(128)",
            fontsize=8, color="#898781", ha="right")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=-2e-3)
    ax.set_xlabel(f"components ablated per token (of {ours['C']}, least "
                  "important first for that token;\ntrue single-ablation "
                  "importance unless the legend says otherwise)",
                  color="#52514e")
    ax.set_ylabel("ΔCE on the token (nats, symlog)", color="#52514e")
    ax.set_title("Per-token oracle minimality: toy induction model",
                 color="#0b0b0b", fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", lw=0.75)
    ax.tick_params(colors="#898781")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e",
              loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path,
                    default=Path(__file__).parent / "out/B_layer_K400_C100_long")
    ap.add_argument("--order", default="global_asc")
    ap.add_argument("--prev_run", type=Path, default=None,
                    help="toy-model run dir of the previous clustering method "
                         "(prev_method.py + ablation_curve.py --components); "
                         "plots the per-token oracle ΔCE comparison instead "
                         "of the Llama-log overlay")
    ap.add_argument("--spec", default="per_example_asc:oracle")
    ap.add_argument("--vpd_run", type=Path, default=None,
                    help="optional third curve: VPD toy run dir")
    args = ap.parse_args()

    ours = json.load(open(args.run / "ablation_curve.json"))
    if args.prev_run is not None:
        prev = json.load(open(args.prev_run / "ablation_curve.json"))
        vpd = (json.load(open(args.vpd_run / "ablation_curve.json"))
               if args.vpd_run else None)
        plot_oracle_pair(ours, prev, args.spec,
                         args.run / "oracle_compare.png", vpd=vpd)
        return
    prev = parse_prev(LOG)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    cur = ours["curves"][args.order]
    ax.plot([100 * r["k"] / ours["C"] for r in cur],
            [r["delta"] for r in cur], lw=2, color="#2a78d6",
            label=f"co-factorization, toy induction "
                  f"(C={ours['C']}, base CE {ours['base_ce']:.1e})")
    ax.plot([100 * r["k"] / prev["C"] for r in prev["points"]],
            [r["delta"] for r in prev["points"]], lw=2, color="#eb6834",
            label=f"previous method (VPD shares), Llama-1B Pile "
                  f"(C={prev['C']}, base CE 0.45)")

    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(bottom=-2e-3)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.text(1, 0, "no ablation (ΔCE = 0)", fontsize=8, color="#898781",
            va="bottom")
    ax.set_xlabel("components ablated, least important first (%)",
                  color="#52514e")
    ax.set_ylabel("ΔCE vs unablated model (nats, symlog)", color="#52514e")
    ax.set_title("Minimality: ablating components least-important-first",
                 color="#0b0b0b", fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", lw=0.75)
    ax.tick_params(colors="#898781")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.legend(frameon=False, fontsize=8, labelcolor="#52514e",
              loc="upper left")
    fig.tight_layout()
    out_png = args.run / "ablation_compare.png"
    fig.savefig(out_png, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
