"""How strongly do parameters inside one component interact?

Every row is the same kind of measurement — a co-location statistic against a
null that preserves the marginals and destroys only the pairing — so the rows
are comparable. Ordered, they are monotone in structural distance: a component
binds a computational UNIT tightly, a head loosely, adjacent layers barely, and
composed circuits not at all.
"""
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
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
STRONG, MID, WEAK = "#2a78d6", "#eb6834", "#898781"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

ci = json.loads((RUN / "component_interactions.json").read_text())
rc = json.loads((RUN / "residual_channels.json").read_text())
vh = json.loads((RUN / "virtual_heads.json").read_text())

ROWS = [
    ("MLP neuron  gate ↔ up\nsame layer, same index", ci["pairs"]["gate-up"]["ratio"], STRONG),
    ("MLP neuron  gate ↔ down", ci["pairs"]["gate-down"]["ratio"], STRONG),
    ("MLP neuron  up ↔ down", ci["pairs"]["up-down"]["ratio"], STRONG),
    ("OV dimension  v ↔ o\nsame head, same index", ci["pairs"]["v-o"]["ratio"], MID),
    ("residual channels\nadjacent layers", rc["by_layer_gap"]["1"]["ratio"], MID),
    ("residual channels\n15 layers apart", rc["by_layer_gap"]["15"]["ratio"], WEAK),
    ("virtual head  K-composition\nacross layers", vh["grouping_test"]["K"]["ratio"], WEAK),
]


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(18.6, 7.6), facecolor="white",
                         gridspec_kw={"width_ratios": [1.5, 1]})
fig.subplots_adjust(top=0.665, bottom=0.115, left=0.175, right=0.985,
                    wspace=0.24)

# ---- left: the hierarchy ----
ax = axes[0]
chrome(ax)
ax.grid(True, axis="x", color=GRID, linewidth=0.9 * S)
y = np.arange(len(ROWS))[::-1]
vals = [r[1] for r in ROWS]
cols = [r[2] for r in ROWS]
ax.barh(y, vals, height=0.62, color=cols, zorder=3)
ax.axvline(1.0, color=INK, ls=(0, (4, 3)), lw=1.5 * S)
for yy, v, c in zip(y, vals, cols):
    ax.annotate(f"{v:.1f}×" if v >= 10 else f"{v:.2f}×",
                xy=(v, yy), xytext=(v * 1.14, yy), va="center",
                fontsize=10.5 * S, fontweight="600", color=c)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9.8 * S, color=INK)
ax.set_xscale("log")
ax.set_xlim(0.55, 460)
ax.set_xticks([1, 3, 10, 30, 100])
ax.set_xticklabels(["1× (chance)", "3×", "10×", "30×", "100×"],
                   fontsize=9.8 * S)
ax.set_xlabel("co-location above a marginal-preserving null", fontsize=11 * S,
              color=INK)
ax.set_title("Interaction falls off with structural distance",
             fontsize=13 * S, color=INK, pad=12)

# ---- right: residual-channel decay ----
ax = axes[1]
chrome(ax)
ax.grid(True, color=GRID, linewidth=0.9 * S)
gaps = sorted(int(k) for k in rc["by_layer_gap"])
rat = [rc["by_layer_gap"][str(k)]["ratio"] for k in gaps]
ax.plot(gaps, rat, color=MID, marker="o", linewidth=2.6 * S,
        markersize=7 * S, markeredgecolor="white", markeredgewidth=1.2 * S)
ax.axhline(1.0, color=INK, ls=(0, (4, 3)), lw=1.5 * S)
ax.annotate("chance", xy=(11.4, 1.015), fontsize=9.5 * S, color=INK)
ax.set_ylim(0.98, 1.60)
ax.set_xlabel("layers between write and read", fontsize=11 * S, color=INK)
ax.set_ylabel("residual-channel overlap ÷ null", fontsize=11 * S, color=INK)
ax.set_title("Cross-layer coupling is real but local", fontsize=13 * S,
             color=INK, pad=12)

fig.suptitle("How do parameters inside one component interact?",
             fontsize=16 * S, color=INK, x=0.055, ha="left", y=0.975)
fig.text(0.055, 0.905,
         "Each bar is a co-location statistic against a null that preserves every marginal and "
         "destroys only the pairing, so they are comparable.\n"
         "A component binds a computational UNIT almost perfectly — in SwiGLU, gate row i, up row i "
         "and down column i are one neuron, and components own\n"
         "them together at 138× chance (89× above a layer-shifted control). Binding then decays: "
         "loosely within a head, barely across adjacent layers,\n"
         "and at exactly chance for the composed head pairs that make a virtual attention head.",
         fontsize=9.8 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/component_interactions{ext}", facecolor="white", **kw)
print("wrote out/component_interactions.png/.pdf/.svg")
for lab, v, _ in ROWS:
    print(f"  {lab.splitlines()[0]:<34} {v:.2f}x")
