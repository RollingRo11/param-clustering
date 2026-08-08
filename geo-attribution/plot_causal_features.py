"""Does gradient×weight predict causal necessity? Yes — and the clustering discards it.

Left  — the first-order attribution dR/dW·W, summed over a component's owned
        entries, against the ablation effect actually measured for that
        component. Rank correlation 0.96.
Right — what happens if you integrate the gradient along the weight-scaling
        path instead. Aggregate completeness improves (Σattr/R goes to ~0.95,
        as the theory says it must) but the per-component ranking gets WORSE,
        because most of that path is a model that does not exist.
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
FO, IG = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

d = json.loads((RUN / "causal_features.json").read_text())
s = d["sample"]
actual = np.array([r["actual"] for r in s])
fo = np.array([r["first_order"] for r in s])
igf = np.array([r["ig_full"] for r in s])
cor = d["correlations"]


def chrome(ax):
    ax.set_facecolor("white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.2), facecolor="white")
fig.subplots_adjust(top=0.63, bottom=0.125, left=0.062, right=0.985,
                    wspace=0.23)

# ---- left: rank-vs-rank, both estimators ----
ax = axes[0]
chrome(ax)
ra = np.argsort(np.argsort(actual))
for vec, col, lab, key in ((fo, FO, "gradient × weight  (first order)",
                            "first_order_grad_x_weight"),
                           (igf, IG, "path-integrated  (16 steps)",
                            f"ig_{d['ig_k']}_full")):
    rp = np.argsort(np.argsort(vec))
    ax.scatter(ra, rp, s=78 * S, color=col, edgecolors="white",
               linewidths=1.0 * S, alpha=0.85, zorder=3,
               label=f"{lab}   ρ = {cor[key]['spearman']:+.2f}")
ax.plot([0, len(s)], [0, len(s)], color=INK, ls=(0, (4, 3)), lw=1.4 * S,
        zorder=2)
ax.annotate("perfect ranking", xy=(len(s) * 0.52, len(s) * 0.44),
            fontsize=9.4 * S, color=INK, rotation=32)
ax.set_xlabel("rank by MEASURED ablation effect", fontsize=10.5 * S, color=INK)
ax.set_ylabel("rank by predicted attribution", fontsize=10.5 * S, color=INK)
ax.set_title("Attribution does predict causal necessity",
             fontsize=12.5 * S, color=INK, pad=11)
ax.legend(loc="upper left", frameon=False, fontsize=9.8 * S)

# ---- right: how the estimator degrades along the path ----
ax = axes[1]
chrome(ax)
names = [("first_order_grad_x_weight", "1\n(gradient)"), ("ig_2", "2"),
         ("ig_4", "4"), ("ig_8", "8"), (f"ig_{d['ig_k']}_full",
                                        f"{d['ig_k']}\n(full path)")]
xs = np.arange(len(names))
sp = [cor[n]["spearman"] for n, _ in names]
comp = [abs(cor[n]["sum_ratio_vs_total"]) for n, _ in names]
ax.bar(xs - 0.19, sp, 0.36, color=FO, zorder=3,
       label="rank correlation with true ablation")
ax.bar(xs + 0.19, comp, 0.36, color=MUTED, zorder=3,
       label="|Σ attributions / total|  (completeness)")
ax.axhline(1.0, color=INK, ls=(0, (4, 3)), lw=1.4 * S)
ax.annotate("perfect", xy=(-0.44, 1.03), fontsize=9.4 * S, color=INK)
for x, v in zip(xs - 0.19, sp):
    ax.annotate(f"{v:.2f}", (x, v), (x, v + 0.035), ha="center",
                fontsize=9.6 * S, fontweight="600", color=FO)
ax.set_xticks(xs)
ax.set_xticklabels([n for _, n in names], fontsize=9.6 * S, color=INK)
ax.set_xlabel("steps integrated along the weight-scaling path",
              fontsize=10.5 * S, color=INK)
ax.set_ylim(0, 1.42)
ax.set_title("Integrating the path helps the SUM, hurts the RANKING",
             fontsize=12.5 * S, color=INK, pad=11)
ax.legend(loc="upper center", frameon=False, fontsize=9.4 * S, ncol=1)

fig.suptitle("Is gradient attribution already causal necessity?",
             fontsize=15.5 * S, color=INK, x=0.062, ha="left", y=0.972)
fig.text(0.062, 0.907,
         "Largely yes — dR/dW·W summed over a component's owned entries ranks components by their true "
         "ablation effect at ρ = 0.96 (n = 96,\n"
         "sampled across the whole mass range; measured by actually deleting each one). So the "
         "information IS in the features. What removes it is the\n"
         "CLUSTERING: streaming_decomposition.py L2-normalises the feature vector before spherical "
         "k-means, and magnitude is exactly the part that\n"
         "encodes 'how much this matters'. What survives is direction — which weights fire together. "
         "The decomposition is co-attribution by construction.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/causal_features{ext}", facecolor="white", **kw)
print("wrote out/causal_features.png/.pdf/.svg")
for n, _ in names:
    print(f"  {n:<26} spearman {cor[n]['spearman']:+.3f}  "
          f"completeness {cor[n]['sum_ratio_vs_total']:+.3f}")
