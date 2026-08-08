"""Runtime interaction: which components fire together, and does it matter?

Left  — the co-activation graph is diffuse, not modular: a typical component
        has ~100 effective partners and most pairs fire near-independently.
Right — but co-firing predicts the SIGN of causal interaction. Among components
        whose ablation is measurable at all, pairs that co-fire above chance are
        superadditive under joint ablation; pairs that never co-fire are
        subadditive, i.e. redundant.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 1.2,
})
S = 1.45
GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
HI, LO, INKC = "#2a78d6", "#eb6834", "#0b0b0b"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

cg = json.loads((RUN / "coactivation_graph.json").read_text())
sp = json.loads((RUN / "synergy_powered.json").read_text())
arr = torch.load(RUN / "coactivation_arrays.pt", weights_only=False)
partners = arr["partners"].numpy()


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(17.2, 7.0), facecolor="white",
                         gridspec_kw={"width_ratios": [1, 1.05]})
fig.subplots_adjust(top=0.635, bottom=0.125, left=0.058, right=0.985,
                    wspace=0.24)

# ---- left: how many partners does a component co-fire with? ----
ax = axes[0]
chrome(ax)
ax.grid(True, color=GRID, linewidth=0.9 * S)
ax.hist(partners, bins=60, color=HI, zorder=3)
med = np.median(partners)
ax.axvline(med, color=LO, lw=2.4 * S)
ax.annotate(f"median {med:.0f} partners", xy=(med + 6, ax.get_ylim()[1] * 0.87),
            fontsize=10 * S, color=LO, fontweight="600")
ax.set_xlabel("effective co-activation partners  (participation ratio)",
              fontsize=10.5 * S, color=INK)
ax.set_ylabel("components", fontsize=10.5 * S, color=INK)
ax.set_title("The runtime graph is diffuse, not modular",
             fontsize=12.5 * S, color=INK, pad=11)
act = cg["active_components_per_token"]["p>0.01"]
ax.annotate(f"{act:.1f} components active per token\n"
            f"median pair fires only "
            f"{cg['lift_over_independence']['median_offdiag']:.2f}× above independence",
            xy=(0.97, 0.62), xycoords="axes fraction", ha="right",
            fontsize=9.6 * S, color=INK2, linespacing=1.5)

# ---- right: does co-firing predict causal interaction? ----
ax = axes[1]
chrome(ax)
ax.grid(True, axis="y", color=GRID, linewidth=0.9 * S)
hi = np.array([r["relative"] for r in sp["pairs"]["high_lift"]])
lo = np.array([r["relative"] for r in sp["pairs"]["low_lift"]])
rng = np.random.default_rng(0)
for i, (v, c) in enumerate(((hi, HI), (lo, LO))):
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               s=125 * S, color=c, edgecolors="white", linewidths=1.2 * S,
               zorder=4)
    ax.plot([i - 0.27, i + 0.27], [v.mean()] * 2, color=c, lw=3.4 * S, zorder=5)
ax.axhline(0, color=INKC, ls=(0, (4, 3)), lw=1.5 * S)
ax.annotate("additive", xy=(1.34, 0.006), fontsize=9.4 * S, color=INKC)
ax.annotate("superadditive\n(need each other)", xy=(0.62, 0.145),
            fontsize=9.6 * S, color=HI, fontweight="600", linespacing=1.4)
ax.annotate("subadditive\n(redundant)", xy=(-0.46, -0.105),
            fontsize=9.6 * S, color=LO, fontweight="600", linespacing=1.4)
ax.set_xticks([0, 1])
ax.set_xticklabels(["co-fire above chance\n(lift 5–15×)",
                    "never co-fire\n(lift ≈ 0)"], fontsize=10 * S, color=INK)
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("relative ablation synergy", fontsize=10.5 * S, color=INK)
ax.set_title("Co-firing predicts the sign of causal interaction",
             fontsize=12.5 * S, color=INK, pad=11)
ax.annotate(f"permutation p = 0.0012\nAUC = 0.90   (n = 9 vs 9)",
            xy=(0.50, 0.055), xycoords="axes fraction", ha="center",
            fontsize=9.8 * S, color=INK2, linespacing=1.5)

fig.suptitle("How components interact at runtime",
             fontsize=15.5 * S, color=INK, x=0.058, ha="left", y=0.972)
fig.text(0.058, 0.905,
         "Co-activation is the frozen fingerprint posterior, the same quantity the streaming "
         "assignment uses. Lift = Coact(a,b) / p(a)p(b) — how much more\n"
         "two components fire together than their marginals predict. Synergy is measured by "
         "ablating whole components: dCE(both) − dCE(a) − dCE(b), relative to\n"
         "the main effects. Both arms are drawn from the SAME pool of components (those whose "
         "single ablation is measurable), so main effects are matched and only\n"
         "the pairing differs. Components that fire together need each other; components that never "
         "fire together partly substitute for each other.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/coactivation{ext}", facecolor="white", **kw)
print("wrote out/coactivation.png/.pdf/.svg")
print(f"  high-lift mean {hi.mean():+.4f}  low-lift mean {lo.mean():+.4f}")
