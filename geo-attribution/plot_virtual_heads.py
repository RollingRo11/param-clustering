"""What cross-layer structure the components actually have.

Three panels, in the order the argument has to be made:

  1  the composition score WORKS   — validated at 2.05x on the real
     previous-token -> induction circuit, and only on K, as theory says
  2  but components do not group composing heads — Q/K/V all at chance
  3  they group heads by ROLE instead — same job, several layers
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
OBS, NULL, ACC = "#2a78d6", "#898781", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

vh = json.loads((RUN / "virtual_heads.json").read_text())
vc = json.loads((RUN / "validate_composition.json").read_text())
fg = json.loads((RUN / "functional_grouping.json").read_text())
arr = torch.load(RUN / "functional_grouping_arrays.pt", weights_only=False)
L, H = arr["L"], arr["H"]
zi = arr["z"]["induction"].numpy()
prof = arr["profiles"].numpy()
share3392 = arr["share_c3392"].numpy().reshape(L, H)
ind_map = prof[:, arr["features"].index("induction")].reshape(L, H)


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.6), facecolor="white",
                         gridspec_kw={"width_ratios": [1.05, 1, 1.15]})
fig.subplots_adjust(top=0.575, bottom=0.125, left=0.05, right=0.985,
                    wspace=0.28)

# ---- panel 1+2: composition, validation vs components ----
ax = axes[0]
chrome(ax)
ax.grid(True, axis="y", color=GRID, linewidth=0.9 * S)
kinds = ["Q", "K", "V"]
val = [vc["tests"][k]["ratio"] for k in kinds]
own = [vh["grouping_test"][k]["ratio"] for k in kinds]
x = np.arange(3)
W = 0.36
ax.bar(x - W / 2, val, W - 0.03, color=ACC, zorder=3,
       label="real circuit: prev-token → induction")
ax.bar(x + W / 2, own, W - 0.03, color=OBS, zorder=3,
       label="heads sharing a component")
ax.axhline(1.0, color=INK, ls=(0, (4, 3)), lw=1.4 * S)
ax.annotate("chance", xy=(-0.48, 0.86), fontsize=9.5 * S, color=INK)
for xx, v in zip(x - W / 2, val):
    ax.annotate(f"{v:.2f}×", (xx, v), (xx, v + 0.04), ha="center",
                fontsize=10 * S, fontweight="600", color=ACC)
for xx, v in zip(x + W / 2, own):
    ax.annotate(f"{v:.2f}×", (xx, v), (xx, v + 0.04), ha="center",
                fontsize=10 * S, fontweight="600", color=OBS)
ax.set_xticks(x)
ax.set_xticklabels([f"{k}-composition" for k in kinds], fontsize=10 * S,
                   color=INK)
ax.set_ylim(0, 2.45)
ax.set_ylabel("composition score ÷ random-pair null", fontsize=10.5 * S,
              color=INK)
ax.set_title("The score works; components don't use it",
             fontsize=12 * S, color=INK, pad=12)
ax.legend(loc="upper right", frameon=False, fontsize=9.2 * S)

# ---- panel 3: role enrichment ----
ax = axes[1]
chrome(ax)
ax.grid(True, axis="y", color=GRID, linewidth=0.9 * S)
ax.hist(zi, bins=70, color=OBS, zorder=3)
ax.axvline(3, color=INK, ls=(0, (4, 3)), lw=1.4 * S)
n3 = fg["enrichment"]["induction"]["n_components_z_gt_3"]
ax.annotate(f"z > 3\n{n3} components\n(≈6 expected)", xy=(3.7, 250),
            fontsize=9.5 * S, color=INK)
ax.annotate("c3392", xy=(13.7, 1.6), xytext=(9.0, 22),
            fontsize=9.6 * S, color=ACC, fontweight="600",
            arrowprops=dict(arrowstyle="->", color=ACC, lw=1.8))
ax.set_yscale("log")
ax.set_xlabel("induction-role enrichment  (z vs permutation null)",
              fontsize=10.5 * S, color=INK)
ax.set_ylabel("components", fontsize=10.5 * S, color=INK)
ax.set_title("Components collect a role across layers",
             fontsize=12 * S, color=INK, pad=12)

# ---- panel 4: c3392's map ----
ax = axes[2]
chrome(ax)
ok = share3392 > 0.005
ax.scatter(ind_map[~ok], share3392[~ok] * 100, s=26 * S, color=GRID,
           edgecolors="none", zorder=2)
ax.scatter(ind_map[ok], share3392[ok] * 100, s=150 * S, color=OBS,
           edgecolors="white", linewidths=1.2 * S, zorder=4)
for l in range(L):
    for h in range(H):
        if share3392[l, h] > 0.04:
            ax.annotate(f"L{l}H{h}", (ind_map[l, h], share3392[l, h] * 100),
                        (6, 6), textcoords="offset points",
                        fontsize=9 * S, color=INK, fontweight="600")
ax.set_xlim(-0.06, 1.16)
ax.set_ylim(-0.9, 18.6)
ax.set_xlabel("head's induction score (attention pattern)",
              fontsize=10.5 * S, color=INK)
ax.set_ylabel("share of c3392's attention mass (%)", fontsize=10.5 * S,
              color=INK)
ax.set_title("c3392: 4 induction heads, 3 layers",
             fontsize=12 * S, color=INK, pad=12)

fig.suptitle("Do components correspond to cross-layer virtual attention heads?",
             fontsize=16 * S, color=INK, x=0.05, ha="left", y=0.975)
fig.text(0.05, 0.905,
         "A virtual head is a PAIR of heads whose circuits multiply through the residual stream. "
         "Scoring that (Elhage et al.) recovers the textbook\n"
         "previous-token → induction circuit at 2.05× chance, and specifically on K-composition — "
         "so the measure is sound. Head pairs that share a\n"
         "component nonetheless compose at chance. What components DO capture is heads playing the "
         "same role in different layers: 360 of 4096 are\n"
         "significantly induction-enriched, and co-owned cross-layer pairs are 14% closer in "
         "behaviour than random pairs.",
         fontsize=9.8 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/virtual_heads{ext}", facecolor="white", **kw)
print("wrote out/virtual_heads.png/.pdf/.svg")
print(f"  validation  Q {val[0]:.2f} K {val[1]:.2f} V {val[2]:.2f}")
print(f"  co-owned    Q {own[0]:.2f} K {own[1]:.2f} V {own[2]:.2f}")
print(f"  induction z>3: {n3}; c3392 rank {fg['c3392_induction_rank']}")
