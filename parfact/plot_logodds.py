"""Does the non-saturating logodds score fix v1's canonical ordering? 
Accuracy view: logp (solid) vs logodds (dash-dot) vs oracle (dotted)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent / "out/curves_v2"
base = {t: json.loads((D / f"curves_{t}.json").read_text())
        for t in ("v1", "v2")}
lo = json.loads((D / "curves_logodds.json").read_text())

fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
for tag, color in (("v1", "#2a78d6"), ("v2", "#8a3ffc")):
    for blob, order, style, lab in (
            (base[tag], "canonical", "-", "canonical (logp)"),
            (lo[tag], "canonical", (0, (5, 2, 1, 2)), "canonical (logodds)"),
            (base[tag], "oracle", (0, (2, 2)), "oracle")):
        cv = blob["curves"][order]
        ax.plot([r["k"] for r in cv], [r["acc"] for r in cv], lw=2,
                color=color, ls=style, label=f"co-fac {tag} — {lab}")
ax.axhline(1 / 128, color="#898781", lw=1, ls=(0, (4, 3)))
ax.set_xlabel("components ablated per token (of 600, least important first)",
              color="#52514e")
ax.set_ylabel("induction accuracy", color="#52514e")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Score ablation: logp vs non-saturating logodds — the v1/v2 "
             "gap is the factorization, not the score", fontsize=10.5,
             color="#0b0b0b")
ax.grid(axis="y", color="#e1e0d9", lw=0.75)
ax.tick_params(colors="#898781")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7.2, loc="lower left", framealpha=0.9)
out = Path(__file__).parent / "figures/keep_topk_logodds_ablation.png"
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
