"""Accuracy companion to the keep-top-k CE plot: fraction of events still
correct vs components ablated. Bounded metric -> no heavy-tail jaggedness."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent / "out/curves_v2"
METHODS = (("co-fac v1 (simplex V)", "#2a78d6", "curves_v1.json"),
           ("co-fac v2 (residual + I-div)", "#8a3ffc", "curves_v2.json"),
           ("SPD", "#1baf7a", "curves_spd.json"))

fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
for name, color, fn in METHODS:
    blob = json.loads((D / fn).read_text())
    for order, style in (("canonical", "-"), ("oracle", (0, (2, 2)))):
        curve = blob["curves"][order]
        ax.plot([r["k"] for r in curve], [r["acc"] for r in curve],
                lw=2, color=color, ls=style, label=f"{name} — {order}")
ax.axhline(1 / 128, color="#898781", lw=1, ls=(0, (4, 3)))
ax.text(600, 1 / 128 + 0.015, "chance (1/128)", fontsize=8,
        color="#898781", ha="right")
ax.set_xlabel("components ablated per token (of 600, least important "
              "first for that token)", color="#52514e")
ax.set_ylabel("induction accuracy (fraction of 2048 events)",
              color="#52514e")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Keep-top-k minimality at C=600 — accuracy view",
             color="#0b0b0b", fontsize=11)
ax.grid(axis="y", color="#e1e0d9", lw=0.75)
ax.tick_params(colors="#898781")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
out = Path(__file__).parent / "figures/keep_topk_acc_v1_v2_spd.png"
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
