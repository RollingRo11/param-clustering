"""Weight-mass distribution across the C=4096 co-fac components (1M-event
fit): rank-ordered share of the model's Frobenius weight mass."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("out/mass_dist.json"))
s = np.sort(np.array(d["fro_share"]))[::-1]
s = np.clip(s, 1e-12, None)
rank = np.arange(1, len(s) + 1)
cum = np.cumsum(s) / s.sum()

fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
TOP = 50
ax.bar(rank[:TOP], 100 * s[:TOP], width=1.0, color="#1f3a5f",
       linewidth=0.4, edgecolor="white")
ax.set_xlim(0.5, TOP + 0.5)
ax.set_ylim(0, 90)
ax.set_xlabel("components (sorted by weight mass)")
ax.set_ylabel("% of model weight mass (Frobenius)")
ax.set_xticks([])
ax.annotate(f"{100 * s[0]:.0f}%", (1, 100 * s[0]),
            textcoords="offset points", xytext=(6, 2), fontsize=11,
            color="#1f3a5f", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/mass_distribution.png", bbox_inches="tight")
print("wrote figures/mass_distribution.png")
