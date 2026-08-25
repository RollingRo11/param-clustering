"""Keep-top-k ablation-curve comparison: v1 vs v2 (residual+I-div) vs SPD.

Solid = canonical per-event importance (|z| for co-fac, causal importance
for SPD); dotted = oracle (true single-component ablation). Reads
out/curves_v2/curves_{v1,v2,spd}.json from curves_v2.py.

    python plot_curves_v2.py
"""
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
uniform = None
for name, color, fn in METHODS:
    blob = json.loads((D / fn).read_text())
    uniform = blob["uniform_ce"]
    for order, style in (("canonical", "-"), ("oracle", (0, (2, 2)))):
        curve = blob["curves"][order]
        ax.plot([r["k"] for r in curve], [r["delta"] for r in curve],
                lw=2, color=color, ls=style, label=f"{name} — {order}")
ax.axhline(0, color="#c3c2b7", lw=1)
ax.axhline(uniform, color="#898781", lw=1, ls=(0, (4, 3)))
ax.text(600, uniform * 0.72, "uniform ln(128)", fontsize=8,
        color="#898781", ha="right")
ax.set_yscale("symlog", linthresh=1e-2)
ax.set_ylim(bottom=-2e-3)
ax.set_xlabel("components ablated per token (of 600, least important "
              "first for that token)", color="#52514e")
ax.set_ylabel("ΔCE on the token (nats, symlog)", color="#52514e")
ax.set_title("Keep-top-k minimality at C=600: v1 vs v2 (background-"
             "retaining) vs SPD", color="#0b0b0b", fontsize=11)
ax.grid(axis="y", color="#e1e0d9", lw=0.75)
ax.tick_params(colors="#898781")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
out = Path(__file__).parent / "figures/keep_topk_v1_v2_spd.png"
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")
