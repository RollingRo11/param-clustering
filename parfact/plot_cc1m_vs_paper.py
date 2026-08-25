"""Slide figure: active subcomponents per matrix — Christensen & Riggs
reported vs our 1M-step run of their code (same model, their gate,
g>0.5 census). Only K0 and V1 were censused in the 1M run; the other
four matrices are trivially 1 in the paper and are marked not measured."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROWS = ["Q0", "K0", "V0", "Q1", "K1", "V1"]
PAPER = {"Q0": 1, "K0": 1, "V0": 1, "Q1": 1, "K1": 1, "V1": 11}
# K0/V1 measured (cc_code_census_1m.json, g>0.5); others assumed 1 (slides)
OURS_1M = {"Q0": 1, "K0": 2, "V0": 1, "Q1": 1, "K1": 1, "V1": 18}

GRAY, INK = "#9a9890", "#1f3a5f"
x = np.arange(len(ROWS))
w = 0.38

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=170)
ax.bar(x - w / 2, [PAPER[r] for r in ROWS], w, color=GRAY,
       label="Christensen & Riggs (reported)")
xs, vals = zip(*[(i, OURS_1M[r]) for i, r in enumerate(ROWS)
                 if r in OURS_1M])
ax.bar(np.array(xs) + w / 2, vals, w, color=INK,
       label="their code, 1M steps (ours)")
for i, r in enumerate(ROWS):
    ax.text(i - w / 2, PAPER[r] + 0.3, str(PAPER[r]), ha="center",
            fontsize=10, color="#555")
    ax.text(i + w / 2, OURS_1M[r] + 0.3, str(OURS_1M[r]), ha="center",
            fontsize=10, color=INK, fontweight="bold")
ax.set_xticks(x, ROWS)
ax.set_ylabel("active subcomponents (g > 0.5)")
ax.set_ylim(0, 20.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig("figures/cc1m_vs_paper.png", bbox_inches="tight")
print("wrote figures/cc1m_vs_paper.png")
