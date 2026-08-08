"""Ablation robustness: CE against the number of components removed.

Three orderings bound what the decomposition can survive. Removing the
lightest components first is the generous case; removing the heaviest first is
the adversarial one; random is what a uniformly-chosen subset does.

Two reference lines matter. ln(128256) = 11.76 is uniform guessing, and the
curves go far ABOVE it — a broken model is confidently wrong, not uninformed.
The right-hand endpoint is the direct embed -> unembed path with all 112
transformer matrices zeroed, which is where every ordering has to meet.
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
COL = {"mass_asc": "#2a78d6", "random": "#898781", "mass_desc": "#eb6834"}
NAME = {"mass_asc": "lightest components first", "random": "random order",
        "mass_desc": "heaviest components first"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

d = json.loads((RUN / "ablation_curve.json").read_text())
C, base, uni = d["C"], d["base_ce"], d["uniform_ce"]


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(17.4, 7.2), facecolor="white")
fig.subplots_adjust(top=0.635, bottom=0.125, left=0.058, right=0.985,
                    wspace=0.20)

for ax, zoom in zip(axes, (False, True)):
    chrome(ax)
    for key in ("mass_asc", "random", "mass_desc"):
        c = d["curves"][key]
        k = np.array([r["k"] for r in c], dtype=float)
        ce = np.array([r["ce"] for r in c])
        m = k > 0
        ax.plot(k[m], ce[m], color=COL[key], marker="o", linewidth=2.5 * S,
                markersize=5.4 * S, markeredgecolor="white",
                markeredgewidth=1.1 * S, label=NAME[key])
    ax.axhline(base, color=INK, ls=(0, (4, 3)), lw=1.4 * S)
    ax.set_xscale("log")
    ax.set_xlim(0.8, C * 1.25)
    ax.set_xlabel("components ablated  (of 4,096)", fontsize=10.5 * S,
                  color=INK)
    if not zoom:
        ax.axhline(uni, color=MUTED, ls=(0, (2, 2)), lw=1.4 * S)
        ax.annotate(f"uniform guessing, ln(V) = {uni:.2f}", xy=(1.1, uni + 2.4),
                    fontsize=9.4 * S, color=MUTED)
        ax.annotate(f"unedited model, {base:.2f}", xy=(1.1, base + 2.4),
                    fontsize=9.4 * S, color=INK)
        ax.set_ylabel("cross-entropy (nats)", fontsize=10.5 * S, color=INK)
        ax.set_title("Full range — every ordering ends at the same place",
                     fontsize=12.5 * S, color=INK, pad=11)
        ax.legend(loc="upper left", frameon=False, fontsize=9.8 * S)
    else:
        ax.set_ylim(base - 0.12, base + 1.15)
        ax.annotate(f"unedited model, {base:.2f}", xy=(1.1, base + 0.03),
                    fontsize=9.4 * S, color=INK)
        ax.set_ylabel("cross-entropy (nats)", fontsize=10.5 * S, color=INK)
        ax.set_title("Zoomed — how much can go before behaviour moves",
                     fontsize=12.5 * S, color=INK, pad=11)
        b = d["components_removable_within"]
        rows = [f"{NAME[k].split(' component')[0].split(' order')[0]:<9} "
                f"{b[k]['0.05']:>5}  {b[k]['0.5']:>5}"
                for k in ("mass_asc", "random", "mass_desc")]
        ax.annotate("removable within  ΔCE 0.05 / 0.50\n" + "\n".join(rows),
                    xy=(0.03, 0.975), xycoords="axes fraction", ha="left",
                    va="top", fontsize=9.4 * S, color=INK2,
                    family="monospace", linespacing=1.55)

fig.suptitle("Ablation robustness of the 4,096-component decomposition — VPD §3.2",
             fontsize=15.5 * S, color=INK, x=0.058, ha="left", y=0.972)
fig.text(0.058, 0.912,
         "VPD's version of this plot is partly a FAITHFULNESS curve: its decomposition approximates the "
         "target's weights, needs a residual Δ-component, and\n"
         "reports CE 2.72–3.02 against a 2.71 target. Ours cannot be — the shares sum to exactly 1 per "
         "weight entry, so K=0 reproduces the model bit-for-bit\n"
         "and there is no reconstruction error to show. This is therefore a pure MINIMALITY curve. Note "
         "the curves rise far past uniform guessing: a model with its\n"
         "components removed is confidently wrong, not merely uninformed. Ablating a set is one "
         "elementwise mask per matrix, so all 90 points take 21 seconds.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/ablation_curve{ext}", facecolor="white", **kw)
print("wrote out/ablation_curve.png/.pdf/.svg")
for k, v in d["components_removable_within"].items():
    print(f"  {k:<10} " + "  ".join(f"ΔCE≤{t}: {n}" for t, n in v.items()))
