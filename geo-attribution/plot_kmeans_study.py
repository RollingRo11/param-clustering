"""Magnitude-weighted spherical k-means vs the current fit: no measurable difference.

The whole pipeline was run 3 times (3 pilot seeds), each fitting 4 variants on
identical features: the current method twice with different k-means init, plus
magnitude-weighted and magnitude-squared-weighted. Comparing methods only means
something if the gap beats the gap between two runs of the SAME method — and it
does not come close.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 1.2,
})
S = 1.45
GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
PLAIN, MAG = "#898781", "#2a78d6"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

files = ["kmeans_study_ctrl.json", "kmeans_study_s1.json", "kmeans_study_s2.json"]
ds = [json.loads((RUN / f).read_text()) for f in files]
GROUP = {"plain": PLAIN, "plain_seed2": PLAIN,
         "magweighted": MAG, "magweighted_sq": MAG}
IS_MAG = {"plain": 0, "plain_seed2": 0, "magweighted": 1, "magweighted_sq": 1}


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.0), facecolor="white")
fig.subplots_adjust(top=0.625, bottom=0.125, left=0.062, right=0.985,
                    wspace=0.23)

# ---- left: the metric, per fit ----
ax = axes[0]
chrome(ax)
rng = np.random.default_rng(0)
pts = {0: [], 1: []}
for d in ds:
    for k, v in d["results"].items():
        pts[IS_MAG[k]].append(v["gini_of_effects"])
for i, (lab, col) in enumerate(((
        "current method\n(L2-normalised)", PLAIN),
        ("magnitude-weighted", MAG))):
    v = np.array(pts[i])
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.13, 0.13, len(v)), v,
               s=150 * S, color=col, edgecolors="white", linewidths=1.3 * S,
               zorder=4)
    ax.plot([i - 0.28, i + 0.28], [v.mean()] * 2, color=col, lw=3.4 * S,
            zorder=5)
    ax.annotate(f"mean {v.mean():.3f}\nsd {v.std(ddof=1):.3f}",
                (i, 0.985), ha="center", va="top", fontsize=9.6 * S,
                color=col, fontweight="600", linespacing=1.4)
ax.set_xticks([0, 1])
ax.set_xticklabels(["current method\n(L2-normalised)", "magnitude-weighted"],
                   fontsize=10.2 * S, color=INK)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0.45, 1.03)
ax.set_ylabel("Gini of single-component ablation effects",
              fontsize=10.5 * S, color=INK)
ax.set_title("Higher = the partition separates 'matters' from 'inert'",
             fontsize=12 * S, color=INK, pad=11)
ax.annotate("Welch t = +0.13   (n = 6 fits each)", xy=(0.5, 0.03),
            xycoords="axes fraction", ha="center", fontsize=9.8 * S,
            color=INK2)

# ---- right: the minimality curves, all fits ----
ax = axes[1]
chrome(ax)
base = ds[0]["base_ce"]
for d in ds:
    for k, v in d["results"].items():
        c = [r for r in v["curve"] if r["k"] > 0]
        ax.plot([r["k"] for r in c], [r["ce"] for r in c],
                color=GROUP[k], lw=2.2 * S, alpha=0.75,
                marker="o", markersize=4.4 * S, markeredgecolor="white",
                markeredgewidth=0.9 * S)
ax.axhline(base, color=INK, ls=(0, (4, 3)), lw=1.4 * S)
ax.annotate(f"unedited, {base:.2f}", xy=(9, base + 0.07), fontsize=9.4 * S,
            color=INK)
ax.set_xscale("log")
ax.set_ylim(base - 0.15, base + 2.2)
ax.set_xlabel("components ablated, lightest first  (of 256)",
              fontsize=10.5 * S, color=INK)
ax.set_ylabel("cross-entropy (nats)", fontsize=10.5 * S, color=INK)
ax.set_title("Every fit sheds exactly 16 of 256 within ΔCE 0.05",
             fontsize=12 * S, color=INK, pad=11)
h = [plt.Line2D([], [], color=PLAIN, lw=3 * S, label="current method"),
     plt.Line2D([], [], color=MAG, lw=3 * S, label="magnitude-weighted")]
ax.legend(handles=h, loc="upper left", frameon=False, fontsize=9.8 * S)

fig.suptitle("Testing magnitude-weighted spherical k-means",
             fontsize=15.5 * S, color=INK, x=0.062, ha="left", y=0.972)
fig.text(0.062, 0.907,
         "The whole pipeline — GIM features, PCA embed, k-means, softpart bank, ablation — run 3 times "
         "on Llama-3.2-1B at C=256 with an 8k-position\n"
         "pilot, fitting 4 variants per run on identical features. The decisive control is running the "
         "CURRENT method twice with different k-means init:\n"
         "that alone swings the Gini from 0.56 to 0.89 and the largest single-component effect from "
         "0.12 to 2.56 nats, a 22× range. Against that,\n"
         "magnitude weighting moves nothing (Welch t = +0.13, +0.10, −0.18 on the three metrics). "
         "Total study: 4 minutes per run.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/kmeans_study{ext}", facecolor="white", **kw)
print("wrote out/kmeans_study.png/.pdf/.svg")
for i, n in ((0, "plain"), (1, "magweighted")):
    v = np.array(pts[i])
    print(f"  {n:<14} gini {v.mean():.3f} ± {v.std(ddof=1):.3f}  (n={len(v)})")
