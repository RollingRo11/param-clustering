"""Clustering parameters-by-tokens instead of tokens-by-parameters.

The attribution matrix is [N tokens x D weight coordinates]. The production fit
clusters its ROWS: tokens that use similar parameters. The transpose clusters
its COLUMNS: parameters used by similar tokens. Crossing that with whether
magnitude survives normalisation gives four arms, run at C=256 on a 8k-position
pilot, three seeds each.

The metric is minimality: how many components can be deleted, lightest first,
before cross-entropy moves by 0.05 nats.
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
ARMS = [
    ("rows_sph", "#0b0b0b", "o", "tokens, normalised\n(current method)"),
    ("rows_euclid", "#898781", "s", "tokens, magnitude kept"),
    ("cols_sph", "#2a78d6", "^", "parameters, normalised"),
    ("cols_euclid", "#eb6834", "D", "parameters, magnitude kept"),
]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

ds = [json.loads((RUN / f"transpose_fine_s{s}.json").read_text())
      for s in (0, 1, 2)]
base = ds[0]["base_ce"]
ks = [c["k"] for c in ds[0]["results"]["rows_sph"]["curve"]]


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(17.4, 7.2), facecolor="white",
                         gridspec_kw={"width_ratios": [1.3, 1]})
fig.subplots_adjust(top=0.625, bottom=0.165, left=0.062, right=0.985,
                    wspace=0.22)

ax = axes[0]
chrome(ax)
for key, col, mk, lab in ARMS:
    y = np.array([[d["results"][key]["curve"][i]["ce"] - base
                   for i in range(len(ks))] for d in ds])
    m = np.array(ks) > 0
    ax.plot(np.array(ks)[m], y.mean(0)[m], color=col, marker=mk,
            lw=2.5 * S, markersize=6 * S, markeredgecolor="white",
            markeredgewidth=1.0 * S, label=lab.replace("\n", " "))
    ax.fill_between(np.array(ks)[m], y.min(0)[m], y.max(0)[m], color=col,
                    alpha=0.15, linewidth=0)
ax.axhline(0.05, color=INK, ls=(0, (4, 3)), lw=1.4 * S)
ax.annotate("ΔCE = 0.05", xy=(4.4, 0.056), fontsize=9.6 * S, color=INK)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(3.5, 100)
ax.set_ylim(1e-4, 1.2)
ax.set_xlabel("components deleted, lightest first  (of 256)",
              fontsize=10.5 * S, color=INK)
ax.set_ylabel("ΔCE (nats)", fontsize=10.5 * S, color=INK)
ax.set_title("How much can be deleted before behaviour moves",
             fontsize=12.5 * S, color=INK, pad=11)
ax.legend(loc="lower right", frameon=False, fontsize=9.4 * S)

ax = axes[1]
chrome(ax)
xs = np.arange(len(ARMS))
for i, (key, col, mk, lab) in enumerate(ARMS):
    v = np.array([d["results"][key]["removable_within_0.05"] for d in ds])
    ax.bar(i, v.mean(), 0.6, color=col, zorder=3)
    ax.scatter(np.full(len(v), i), v, s=95 * S, color="white",
               edgecolors=INK, linewidths=1.5 * S, zorder=5)
    ax.annotate(f"{v.mean():.0f}", (i, v.mean()), (i, v.mean() + 1.4),
                ha="center", fontsize=12 * S, fontweight="600", color=col)
ax.set_xticks(xs)
ax.set_xticklabels(["tokens\nnormalised\n(current)", "tokens\nmagnitude\nkept",
                    "parameters\nnormalised", "parameters\nmagnitude\nkept"],
                   fontsize=9.4 * S, color=INK, linespacing=1.35)
ax.set_ylim(0, 56)
ax.set_ylabel("components removable within ΔCE 0.05",
              fontsize=10.5 * S, color=INK)
ax.set_title("Transposing the clustering more than doubles it",
             fontsize=12.5 * S, color=INK, pad=11)
ax.annotate("white dots = the 3 seeds", xy=(0.97, 0.95),
            xycoords="axes fraction", ha="right", fontsize=9.4 * S,
            color=INK2)

fig.suptitle("Clustering parameters by tokens, instead of tokens by parameters",
             fontsize=15 * S, color=INK, x=0.062, ha="left", y=0.972)
fig.text(0.062, 0.907,
         "The attribution matrix is [N tokens × D weight coordinates]. The production fit clusters its "
         "ROWS — tokens that use similar parameters — and\n"
         "derives weight ownership from that. The transpose clusters its COLUMNS: parameters used by "
         "similar tokens. Crossed with whether magnitude\n"
         "survives normalisation, that gives four arms, each run through the full pipeline (GIM "
         "features → k-means → softpart bank → ablation) at C=256\n"
         "on a 8k-position pilot, three seeds. The transposed k-means costs 0.1–0.5 s, cheaper than "
         "the row k-means it replaces; the bank build dominates at ~40 s.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/transpose_study{ext}", facecolor="white", **kw)
print("wrote out/transpose_study.png/.pdf/.svg")
for key, _, _, lab in ARMS:
    v = [d["results"][key]["removable_within_0.05"] for d in ds]
    print(f"  {key:<13} removable {v} mean {np.mean(v):.1f}")
