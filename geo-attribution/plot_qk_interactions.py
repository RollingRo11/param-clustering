"""VPD Section 4.3 ported: how components interact through the QK circuit.

Three levels of the same question, left to right, weakest commitment to
strongest: what the weights allow, what co-activation selects, what ablation
actually does.
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
SAME, CROSS, ACC = "#2a78d6", "#c3c2b7", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

qk = json.loads((RUN / "qk_interactions.json").read_text())
dd = json.loads((RUN / "qk_data_dependent.json").read_text())
ps = json.loads((RUN / "pair_synergy_dedup.json").read_text())
ph = torch.load(RUN / "qk_data_dependent_perhead.pt", weights_only=False)["1"]


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.8), facecolor="white")
fig.subplots_adjust(top=0.60, bottom=0.215, left=0.055, right=0.985,
                    wspace=0.30)

# ---- 1: static, where the QK interaction mass sits ----
ax = axes[0]
chrome(ax)
ax.grid(True, axis="y", color=GRID, linewidth=0.9 * S)
d = qk["diagonal_mass_fraction_tau1"]
r = dd["results"]["1"]
bars = [("weights\n(static)", d["mean"]), ("on text\n(data-dependent)",
                                           r["data_dependent_diagonal_fraction"])]
x = np.arange(2)
ax.bar(x, [1 - b for _, b in bars], 0.55, color=CROSS, zorder=3,
       label="between DIFFERENT components")
ax.bar(x, [b for _, b in bars], 0.55, bottom=[1 - b for _, b in bars],
       color=SAME, zorder=3, label="within the same component")
for xx, (_, b) in zip(x, bars):
    ax.annotate(f"{100 * b:.1f}%", (xx, 1.0), (xx, 1.012), ha="center",
                fontsize=11 * S, fontweight="600", color=SAME)
    ax.annotate(f"{100 * (1 - b):.1f}%", (xx, 0.5), ha="center", va="center",
                fontsize=11.5 * S, fontweight="600", color=INK2)
ax.set_xticks(x)
ax.set_xticklabels([n for n, _ in bars], fontsize=10 * S, color=INK)
ax.set_ylim(0, 1.09)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_ylabel("share of QK interaction mass", fontsize=10.5 * S, color=INK)
ax.set_title("Circuits span components", fontsize=12.5 * S, color=INK, pad=11)
ax.legend(loc="upper center", frameon=False, fontsize=9.2 * S,
          bbox_to_anchor=(0.5, -0.135), ncol=1)

# ---- 2: per-head amplification ----
ax = axes[1]
chrome(ax)
ax.grid(True, color=GRID, linewidth=0.9 * S)
rat = np.array(ph["ratio"])
rat = rat[rat > 0]
ax.hist(np.log10(rat), bins=55, color=SAME, zorder=3)
ax.axvline(0, color=INK, ls=(0, (4, 3)), lw=1.5 * S)
ax.annotate("no change", xy=(0.08, ax.get_ylim()[1] * 0.88),
            fontsize=9.2 * S, color=INK)
med = np.median(rat)
ax.axvline(np.log10(med), color=ACC, lw=2.0 * S)
ax.annotate(f"median {med:.2f}×", xy=(np.log10(med) - 0.12,
                                      ax.get_ylim()[1] * 0.62),
            fontsize=9.4 * S, color=ACC, fontweight="600", ha="right")
ax.set_xticks([-3, -2, -1, 0, 1, 2])
ax.set_xticklabels(["0.001×", "0.01×", "0.1×", "1×", "10×", "100×"],
                   fontsize=9.5 * S)
ax.set_xlabel("co-activation's effect on within-component share",
              fontsize=10.5 * S, color=INK)
ax.set_ylabel("attention heads", fontsize=10.5 * S, color=INK)
ax.set_title(f"Only {100 * dd['results']['1']['frac_heads_amplified']:.0f}% of "
             f"heads concentrate", fontsize=12.5 * S, color=INK, pad=11)

# ---- 3: causal synergy ----
ax = axes[2]
chrome(ax)
ax.grid(True, axis="y", color=GRID, linewidth=0.9 * S)
top = np.array([p["synergy"] for p in ps["pairs"]["top"]]) * 1000
ctl = np.array([p["synergy"] for p in ps["pairs"]["control"]]) * 1000
rng = np.random.default_rng(0)
for i, (v, c, lab) in enumerate(((top, ACC, "strongest QK pairs"),
                                 (ctl, MUTED, "random pairs"))):
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.11, 0.11, len(v)), v,
               s=95 * S, color=c, edgecolors="white", linewidths=1.1 * S,
               zorder=4, label=lab)
    ax.plot([i - 0.26, i + 0.26], [v.mean()] * 2, color=c, lw=3.2 * S,
            zorder=5)
ax.axhline(0, color=INK, ls=(0, (4, 3)), lw=1.5 * S)
ax.annotate("additive", xy=(-0.45, 0.07), fontsize=9.2 * S, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["strongest\nQK pairs", "random\npairs"],
                   fontsize=10 * S, color=INK)
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("ablation synergy  (millinats)", fontsize=10.5 * S, color=INK)
ax.set_title(f"No causal synergy  (t = {ps['summary']['top_minus_control_t']:+.2f})",
             fontsize=12.5 * S, color=INK, pad=11)

fig.suptitle("Porting VPD §4.3: do components interact through the QK circuit?",
             fontsize=15.5 * S, color=INK, x=0.055, ha="left", y=0.975)
fig.text(0.055, 0.915,
         "VPD decomposes W_QK = Σ_{c,c'} over subcomponent PAIRS. Their components are rank-1 so that "
         "term is a scalar; ours are not, but the\n"
         "decomposition is still exact, and the pair strength generalises to "
         "‖A_Q,c^T R_τ A_K,c'‖_F = √tr(R^T G_Q R G_K), which reduces to VPD's\n"
         "quantity at rank 1. Measured over all 16×32 heads and 13 RoPE offsets: the interaction is "
         "overwhelmingly BETWEEN components, matching\n"
         "VPD's finding that circuits are not contained in one subcomponent — but ablation finds no "
         "synergy, so the pairing is not causally load-bearing.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/qk_interactions{ext}", facecolor="white", **kw)
print("wrote out/qk_interactions.png/.pdf/.svg")
