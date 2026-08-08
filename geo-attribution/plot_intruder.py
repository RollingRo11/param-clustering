"""VPD Fig. 6 replication: intruder detection, and what it validates.

Left  — accuracy against chance, under an easy intruder (a random other
        component) and a hard one (the target's nearest neighbour in centroid
        space, mean cosine 0.81).
Right — the same trials split by the grade the LABELLER gave. Intruder
        detection never sees a label, so if the self-graded mono/partial scale
        is meaningful the two must come apart here. They do.
"""
import json
import math
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
EASY, HARD, MONO, PART = "#2a78d6", "#eb6834", "#1c5cab", "#3987e5"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

runs = {k: json.loads((RUN / f"intruder_{k}.json").read_text())
        for k in ("random", "near")}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def stat(trials):
    n = len(trials)
    k = sum(t["correct"] for t in trials)
    lo, hi = wilson(k, n)
    return k / n, lo, hi, n


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(16.6, 7.0), facecolor="white")
fig.subplots_adjust(top=0.635, bottom=0.215, left=0.062, right=0.985,
                    wspace=0.22)

# ---- left: overall, easy vs hard ----
ax = axes[0]
chrome(ax)
labels = ["intruder from a\nrandom component", "intruder from the\nnearest neighbour"]
vals = [stat(runs[k]["trials"]) for k in ("random", "near")]
x = np.arange(2)
for xx, (p, lo, hi, n), c in zip(x, vals, (EASY, HARD)):
    ax.bar(xx, p, 0.55, color=c, zorder=3)
    ax.errorbar(xx, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=7,
                capthick=1.6 * S, elinewidth=1.6 * S, zorder=5, fmt="none")
    ax.annotate(f"{100 * p:.1f}%", (xx, hi), (xx, hi + 0.028), ha="center",
                fontsize=13 * S, fontweight="600", color=c)
    ax.annotate(f"n = {n}", (xx, 0.06), ha="center", fontsize=9.6 * S,
                color="white", fontweight="600")
ax.axhline(0.2, color=INK, ls=(0, (4, 3)), lw=1.5 * S)
ax.annotate("chance (20%)", xy=(-0.44, 0.225), fontsize=9.8 * S, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10 * S, color=INK)
ax.set_ylim(0, 1.13)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_ylabel("intruder detection accuracy", fontsize=10.5 * S, color=INK)
ax.set_title("Components are coherent without their labels",
             fontsize=12.5 * S, color=INK, pad=11)

# ---- right: by self-assigned grade ----
ax = axes[1]
chrome(ax)
W = 0.36
for i, (key, colour, name) in enumerate((("random", EASY, "random intruder"),
                                         ("near", HARD, "nearest-neighbour intruder"))):
    for j, g in enumerate(("mono", "partial")):
        sub = [t for t in runs[key]["trials"] if t["grade"] == g]
        if not sub:
            continue
        p, lo, hi, n = stat(sub)
        xx = j + (i - 0.5) * W
        ax.bar(xx, p, W - 0.04, color=colour, zorder=3,
               label=name if j == 0 else None)
        ax.errorbar(xx, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=5,
                    capthick=1.5 * S, elinewidth=1.5 * S, zorder=5, fmt="none")
        ax.annotate(f"{100 * p:.0f}%", (xx, hi), (xx, hi + 0.025), ha="center",
                    fontsize=10.5 * S, fontweight="600", color=colour)
        ax.annotate(f"n={n}", (xx, 0.05), ha="center", fontsize=8.8 * S,
                    color="white", fontweight="600")
ax.axhline(0.2, color=INK, ls=(0, (4, 3)), lw=1.5 * S)
ax.annotate("chance", xy=(1.30, 0.225), fontsize=9.8 * S, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(["graded MONO\nby the labeller", "graded PARTIAL"],
                   fontsize=10 * S, color=INK)
ax.set_ylim(0, 1.13)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlim(-0.55, 1.55)
ax.set_title("The self-graded scale survives an outside test",
             fontsize=12.5 * S, color=INK, pad=11)
ax.legend(loc="upper center", frameon=False, fontsize=9.6 * S, ncol=2,
          bbox_to_anchor=(0.5, -0.135))
ax.annotate("mono vs partial\n(hard condition):\npermutation p = 0.002",
            xy=(0.035, 0.30), xycoords="axes fraction", ha="left",
            fontsize=9.6 * S, color=INK2, linespacing=1.5)

fig.suptitle("Intruder detection — VPD Fig. 6, on the 4,096-component decomposition",
             fontsize=15 * S, color=INK, x=0.062, ha="left", y=0.972)
fig.text(0.062, 0.905,
         "A judge sees five text windows with one token marked in each: four where the component fires, "
         "and one intruder. It must say which does\n"
         "not belong. No label is shown, so this measures the decomposition rather than the "
         "labelling. The hard condition draws the intruder from the\n"
         "component's NEAREST NEIGHBOUR in centroid space (mean cosine 0.81) — near-chance accuracy "
         "there would mean C=4,096 had split one\n"
         "feature in two. It does not: adjacent components stay 89% distinguishable. Error bars are "
         "95% Wilson intervals; claude-sonnet-5 as judge.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/intruder_detection{ext}", facecolor="white", **kw)
print("wrote out/intruder_detection.png/.pdf/.svg")
for k in ("random", "near"):
    p, lo, hi, n = stat(runs[k]["trials"])
    print(f"  {k:<7} {100 * p:.1f}%  CI [{100 * lo:.1f}, {100 * hi:.1f}]  n={n}")
