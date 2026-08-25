"""VPD per-subcomponent weight-mass distribution (38,912 rank-1
subcomponents), sorted, log-y — companion to the co-fac version."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("out/vpd_mass_dist.json"))
s = np.sort(np.array(d["fro_share"]))[::-1]
s = np.clip(s, 1e-12, None)
rank = np.arange(1, len(s) + 1)

fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
ax.bar(rank, 100 * s, width=1.0, color="#8c2f2f", linewidth=0)
ax.set_yscale("log")
ax.set_xlim(0, len(s) + 1)
ax.set_ylim(1e-8, 200)
ax.set_xlabel("VPD subcomponents (sorted by weight mass)")
ax.set_ylabel("% of model weight mass (Frobenius)")
ax.set_xticks([])
ax.annotate(f"{100 * s[0]:.2g}%", (1, 100 * s[0]),
            textcoords="offset points", xytext=(6, 2), fontsize=11,
            color="#8c2f2f", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/vpd_mass_distribution.png", bbox_inches="tight")
print("wrote figures/vpd_mass_distribution.png")
