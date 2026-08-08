"""Cluster mass across the 4,096 components.

Two different quantities, deliberately shown on one scale:
  * initial clustering  - spherical k-means was fit on a 262,144-position
    pilot; `pilot_cluster_sizes` is how those rows split across centroids.
  * corpus assignment   - the frozen centroids then assign every position in
    the streamed corpus. k-means balances the pilot; real token frequencies
    are not balanced, so this is the skewed one.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")

CORPUS_FILE = "evidence_prop1b_1B.json"    # exhaustive 1B-position sweep
model = torch.load(RUN / "stream_model.pt", weights_only=True,
                   map_location="cpu")
pilot = model["pilot_cluster_sizes"].double()
pilot_n = int(pilot.sum())
pilot = (pilot / pilot_n).numpy()

ev = json.loads((RUN / CORPUS_FILE).read_text())
corpus = np.array([ev[str(c)]["fire_rate"] for c in range(len(ev))])
corpus_n = 1_000_017_920
C = len(corpus)

PILOT_C, CORP_C = "#eb6834", "#2a78d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
UNIFORM = 1.0 / C

fig = plt.figure(figsize=(13.0, 8.8), facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.46,
                      top=0.815, bottom=0.075, left=0.075, right=0.985)


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9)


x = np.arange(C)
YMAX = 100 * max(corpus.max(), pilot.max()) * 1.12
for row, (vals, color, title, n) in enumerate((
        (pilot, PILOT_C,
         "Initial clustering — how the 262,144 pilot positions split across centroids", pilot_n),
        (corpus, CORP_C,
         f"Corpus assignment — fraction of all {corpus_n/1e9:.2f}B token positions "
         f"landing in each component", corpus_n))):
    ax = fig.add_subplot(gs[row])
    chrome(ax)
    ax.bar(x, 100 * vals, width=1.0, linewidth=0, color=color, rasterized=True)
    ax.axhline(100 * UNIFORM, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
    ax.annotate(f"uniform = 1/4096 = {100*UNIFORM:.4f}%",
                xy=(C * 0.997, 100 * UNIFORM), fontsize=9, color=INK2,
                ha="right", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.6))
    ax.set_xlim(0, C)
    ax.set_ylim(0, YMAX)
    ax.set_ylabel("% of positions", fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=11.5, color=INK, pad=7, loc="left")
    ax.set_xticks([0, 1024, 2048, 3072, 4096])
    if row == 1:
        ax.set_xlabel("component (index order, 4,096 bars, no gaps)",
                      fontsize=10.5, color=INK)
    top10 = np.sort(vals)[::-1][:C // 10].sum()
    ax.annotate(f"max {100*vals.max():.3f}%   median {100*np.median(vals):.4f}%   "
                f"max/median {vals.max()/np.median(vals):.1f}×   "
                f"top 10% of components hold {100*top10:.1f}% of mass",
                xy=(0.006, 0.90), xycoords="axes fraction", fontsize=9.5,
                color=color, ha="left", va="top", fontweight="600")

# ---- concentration ----
ax = fig.add_subplot(gs[2])
chrome(ax)
ax.grid(True, axis="x", color=GRID, linewidth=0.9)
frac = 100 * np.arange(1, C + 1) / C
for vals, color, label in ((pilot, PILOT_C, "initial clustering (pilot)"),
                           (corpus, CORP_C, "corpus assignment")):
    ax.plot(frac, 100 * np.cumsum(np.sort(vals)[::-1]), color=color,
            linewidth=2.0, label=label)
ax.plot([0, 100], [0, 100], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1,
        label="perfectly uniform")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("components, ranked by mass (%)", fontsize=10.5, color=INK)
ax.set_ylabel("cumulative % of positions", fontsize=10.5, color=INK)
ax.set_title("Concentration — how far the mass departs from uniform",
             fontsize=11.5, color=INK, pad=7, loc="left")
ax.legend(frameon=False, fontsize=10, loc="lower right")
for vals, color in ((pilot, PILOT_C), (corpus, CORP_C)):
    y = 100 * np.cumsum(np.sort(vals)[::-1])[C // 10 - 1]
    ax.plot([10], [y], marker="o", color=color, markersize=7,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate(f"{y:.0f}%", xy=(10, y), xytext=(12.5, y - 3.5), fontsize=10,
                color=color, fontweight="600")

fig.suptitle("Cluster mass across the 4,096 parameter components — Llama-3.2-1B, C=4096",
             fontsize=15, color=INK, x=0.075, ha="left", y=0.975)
fig.text(0.075, 0.932,
         "Every token position is assigned to exactly one component (nearest centroid), so these bars sum to 100%. "
         "k-means balances the pilot it is fit on;\nthe corpus is 2.9× more skewed — but no component runs away, "
         "and none is empty.",
         fontsize=10, color=MUTED, ha="left", va="top", linespacing=1.5)

out = GEO / "out/cluster_mass.png"
fig.savefig(out, dpi=170, facecolor="white")
print("wrote", out)
for name, vals, n in (("pilot", pilot, pilot_n), ("corpus", corpus, corpus_n)):
    s = np.sort(vals)[::-1]
    print(f"{name:<7} n={n:>12,}  max {vals.max():.3e}  median {np.median(vals):.3e}  "
          f"min {vals.min():.3e}  max/med {vals.max()/np.median(vals):.1f}  "
          f"top10% {100*s[:C//10].sum():.1f}%  empty {(vals == 0).sum()}")
