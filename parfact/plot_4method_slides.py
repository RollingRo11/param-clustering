"""Slide figure: ablation accuracy curves for co-fac v1 / v2 /
v2+U-simplex / SPD. Two panels: canonical (left, solid), oracle (right,
dotted)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent / "out/curves_v2"
METHODS = (("co-fac v1", "#2a78d6", "curves_v1.json"),
           ("co-fac v2", "#8a3ffc", "curves_v2.json"),
           ("co-fac v2 + U-simplex", "#d6572a", "curves_v2_usimplex.json"),
           ("SPD (their code, 1M steps)", "#1baf7a", "curves_spd1m.json"))

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=170, sharey=True)
for ax, order, style, title in ((axes[0], "canonical", "-", "canonical"),
                                (axes[1], "oracle", (0, (2, 2)), "oracle")):
    for name, color, fn in METHODS:
        curve = json.loads((D / fn).read_text())["curves"][order]
        ax.plot([r["k"] for r in curve], [r["acc"] for r in curve],
                lw=2.2, color=color, ls=style, label=name)
    ax.axhline(1 / 128, color="#898781", lw=1, ls=(0, (4, 3)))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("components ablated per token (of 600)")
    ax.set_xlim(0, 600)
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].text(596, 1 / 128 + 0.02, "chance", fontsize=9, color="#898781",
             ha="right")
axes[0].set_ylabel("induction accuracy")
axes[0].legend(frameon=False, fontsize=10, loc="lower left")
fig.tight_layout()
fig.savefig("figures/ablation_4methods_panels.png", bbox_inches="tight")
print("wrote figures/ablation_4methods_panels.png")
