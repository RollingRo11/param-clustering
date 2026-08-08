"""Ablation curve on VPD's 67M target.

x = components deleted (lightest first), y = cross-entropy. Left panel is the
full range, right panel zooms on where each arm crosses DCE 0.05.

Crossing markers are read off the JSON, not hardcoded, so the same script
serves C=256 and C=4096.

    python3.12 plot_transpose67_curve.py --json out/transpose67.json \
        --out out/transpose67_curve --note "..."
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 1.2,
})
S = 1.45
GEO = Path("/workspace/param-clustering/geo-attribution")
ARMS = [
    ("rows_sph", "#0b0b0b", "o", "tokens, normalised  (current method)"),
    ("rows_euclid", "#2a78d6", "s", "tokens, magnitude kept"),
    ("cols_sph", "#eb6834", "^", "parameters, normalised  (transpose)"),
    ("cols_euclid", "#898781", "D", "parameters, magnitude kept"),
]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

ap = argparse.ArgumentParser()
ap.add_argument("--json", type=Path, default=GEO / "out/transpose67.json")
ap.add_argument("--out", type=Path, default=GEO / "out/transpose67_curve")
ap.add_argument("--title", default="Ablation curve on VPD's 67M Pile target")
ap.add_argument("--note", default="")
ap.add_argument("--logx", action="store_true")
ap.add_argument("--zoom_hi", type=float, default=0.0,
                help="x limit of the zoom panel; 0 = auto")
args = ap.parse_args()

d = json.loads(args.json.read_text())
base, C, N = d["base_ce"], d["C"], d["N"]
cross = {k: d["results"][k]["removable_within_0.05"] for k, _, _, _ in ARMS}
zoom_hi = args.zoom_hi or max(4, max(cross.values()) * 1.6)


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=10 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.2), facecolor="white")
fig.subplots_adjust(top=0.655, bottom=0.125, left=0.062, right=0.985,
                    wspace=0.20)

for ax, zoom in zip(axes, (False, True)):
    chrome(ax)
    for key, col, mk, lab in ARMS:
        c = d["results"][key]["curve"]
        k = np.array([r["k"] for r in c], float)
        ce = np.array([r["ce"] for r in c])
        if zoom:
            m = k <= zoom_hi
            k, ce = k[m], ce[m]
        elif args.logx:
            k, ce = k[k > 0], ce[k > 0]
        # thin the markers: at C=4096 the grid has ~85 points per arm
        ev = max(1, len(k) // 16)
        ax.plot(k, ce, color=col, marker=mk, lw=2.6 * S,
                markersize=(6.2 if len(k) < 40 else 5.4) * S, markevery=ev,
                markeredgecolor="white", markeredgewidth=1.1 * S, label=lab)
    ax.axhline(base, color=INK, ls=(0, (4, 3)), lw=1.5 * S)
    if not zoom:
        top = max(r["ce"] for key, _, _, _ in ARMS
                  for r in d["results"][key]["curve"])
        # log y: the K=C endpoint is ~30x base and would flatten everything else
        logy = top > 8 * base
        if args.logx:
            ax.set_xscale("log")
            ax.set_xlim(3.0, C * 1.15)
            xa = 4.0
        else:
            ax.set_xlim(-C * 0.025, C * 1.03)
            xa = C * 0.015
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(base * 0.90, top * 1.35)
            ya = base * 0.985
            ax.set_yticks([3, 5, 10, 20, 40, 80])
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
            ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.set_ylim(base - 0.05 * top, top * 1.06)
            ya = base + 0.04 * top
        # park it below the baseline on the right: nothing ever goes there
        ax.annotate(f"unedited model, CE {base:.2f}",
                    xy=(ax.get_xlim()[1], ya), ha="right", va="top",
                    fontsize=10 * S, color=INK)
        ax.set_title("Full range" + ("  (log y)" if logy else ""),
                     fontsize=13 * S, color=INK, pad=11)
        ax.legend(loc="upper left", frameon=False, fontsize=10 * S)
    else:
        ax.axhline(base + 0.05, color=MUTED, ls=(0, (2, 2)), lw=1.5 * S)
        xa = zoom_hi * 0.02
        # right edge: the crossing labels all live left of it, curves above it
        ax.annotate("ΔCE = 0.05", xy=(zoom_hi * 0.985, base + 0.056),
                    ha="right", fontsize=9.8 * S, color=MUTED)
        ax.annotate(f"unedited, {base:.3f}", xy=(xa, base - 0.024),
                    fontsize=9.8 * S, color=INK)
        # stagger crossing labels vertically when they'd collide horizontally
        order = sorted(ARMS, key=lambda a: cross[a[0]])
        placed = []
        for key, col, _, _ in order:
            n = cross[key]
            ax.plot([n, n], [base - 0.03, base + 0.05], color=col,
                    lw=2.2 * S, alpha=0.55)
            tier = 0
            while any(abs(n - pn) < zoom_hi * 0.13 and t == tier
                      for pn, t in placed):
                tier += 1
            placed.append((n, tier))
            ax.annotate(f"{n}", xy=(n, base + 0.060 + 0.024 * tier),
                        ha="center", fontsize=10.5 * S, color=col,
                        fontweight="600")
        ax.set_ylim(base - 0.040, base + 0.30)
        ax.set_xlim(-zoom_hi * 0.02, zoom_hi)
        ax.set_title("Zoomed — where each arm crosses ΔCE 0.05",
                     fontsize=13 * S, color=INK, pad=11)
    ax.set_xlabel(f"components deleted, lightest first  (of {C})",
                  fontsize=11 * S, color=INK)
    ax.set_ylabel("cross-entropy (nats)", fontsize=11 * S, color=INK)

fig.suptitle(args.title, fontsize=15.5 * S, color=INK, x=0.062, ha="left",
             y=0.972)
fig.text(0.062, 0.912, args.note, fontsize=9.8 * S, color=MUTED, ha="left",
         va="top", linespacing=1.62)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(f"{args.out}{ext}", facecolor="white", **kw)
print(f"wrote {args.out}.png/.pdf/.svg   C={C} N={N} base={base:.4f}")
for key, _, _, lab in ARMS:
    r = d["results"][key]
    at = min((c for c in r["curve"] if c["k"] >= 32), key=lambda c: c["k"])
    print(f"  {key:<13} removable {r['removable_within_0.05']:>4}  "
          f"CE@{at['k']} {at['ce']:.4f}  gini {r['gini_of_effects']:.3f}")
