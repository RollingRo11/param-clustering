"""Hazardous-capability unlearning: how much comes out, and what it costs.

Two things have to be shown together or the figure lies. Driving WMDP-bio to
chance is easy if you are allowed to break the model; leaving MMLU untouched is
easy if you barely edit. So the left panel is the whole (lr, lambda, step)
cloud plotted as removal against collateral, and the right panel is the single
operating point that keeps general knowledge intact.

Unlike the German edit, no point sits in the free corner. That is the result.
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
K1, K4 = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CHANCE = 0.25

runs = [("k=1  —  c3203, 112 scalars", "wmdp_edit_c3203.json", K1, "o"),
        ("k=4  —  top-4 components, 448 scalars", "wmdp_edit_k4.json", K4, "s")]
data = {f: json.loads((RUN / f).read_text()) for _, f, _, _ in runs}
base = data["wmdp_edit_k4.json"]["baseline"]
B_BIO, B_MMLU = base["bio_eval"]["acc"], base["mmlu"]["acc"]

# the operating point: most bio removed while MMLU stays within half a point
MMLU_BUDGET = 0.005


def pick(d):
    ok = [p for p in d["points"]
          if p["mcq"]["mmlu"]["acc"] >= B_MMLU - MMLU_BUDGET]
    return min(ok, key=lambda p: p["mcq"]["bio_eval"]["acc"])


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(17.4, 7.4), facecolor="white",
                         gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(top=0.70, bottom=0.105, left=0.055, right=0.985,
                    wspace=0.22)

# ---- left: the whole frontier ----
ax = axes[0]
chrome(ax)
for label, f, color, mk in runs:
    pts = data[f]["points"]
    x = np.array([B_MMLU - p["mcq"]["mmlu"]["acc"] for p in pts]) * 100
    y = np.array([B_BIO - p["mcq"]["bio_eval"]["acc"] for p in pts]) * 100
    ax.scatter(x, y, s=95 * S, marker=mk, facecolors=color,
               edgecolors="white", linewidths=1.1 * S, alpha=0.88, zorder=3,
               label=label)
head = (B_BIO - CHANCE) * 100
ax.axhline(head, color=MUTED, ls=(0, (4, 3)), lw=1.3 * S)
ax.annotate("all removable hazardous knowledge\n(WMDP-bio at chance)",
            xy=(0.35, head - 0.7), fontsize=9 * S, color=MUTED, va="top")
ax.axvline(0, color=AXIS, lw=1.1 * S)
ax.set_xlabel("collateral — MMLU accuracy lost (points)", fontsize=11 * S,
              color=INK)
ax.set_ylabel("hazardous knowledge removed\nWMDP-bio accuracy lost (points)",
              fontsize=11 * S, color=INK)
ax.set_title("Every configuration swept", fontsize=13 * S, color=INK, pad=10)
ax.legend(loc="lower right", frameon=False, fontsize=10 * S)
ax.set_xlim(-2.6, 13.5)
ax.set_ylim(-1, head * 1.14)

# ---- right: the operating point that keeps MMLU ----
ax = axes[1]
chrome(ax)
BARS = [("WMDP-bio\n(held out)", "bio_eval"), ("MMLU", "mmlu"),
        ("WMDP-cyber", "cyber_eval")]
W = 0.26
xs = np.arange(len(BARS))
series = [("unedited", MUTED, None)] + [(l.split("—")[0].strip(), c, f)
                                        for l, f, c, _ in runs]
for i, (name, color, f) in enumerate(series):
    if f is None:
        v = [base[k]["acc"] for _, k in BARS]
    else:
        p = pick(data[f])
        v = [p["mcq"][k]["acc"] for _, k in BARS]
    ax.bar(xs + (i - 1) * W, v, width=W - 0.035, color=color, zorder=3,
           label=name)
    for x, y in zip(xs + (i - 1) * W, v):
        ax.annotate(f"{100 * y:.1f}", xy=(x, y), xytext=(x, y + 0.012),
                    ha="center", fontsize=9.2 * S, fontweight="600",
                    color=color)
ax.axhline(CHANCE, color=INK, ls=(0, (4, 3)), lw=1.3 * S)
ax.annotate("chance (25%)", xy=(-0.45, CHANCE + 0.008), fontsize=9 * S,
            color=INK)
ax.set_xticks(xs)
ax.set_xticklabels([n for n, _ in BARS], fontsize=10 * S, color=INK)
ax.set_ylim(0, 0.55)
ax.set_yticklabels([f"{int(100 * t)}%" for t in ax.get_yticks()])
ax.set_ylabel("accuracy", fontsize=11 * S, color=INK)
ax.set_title(f"At the operating point that keeps MMLU (≤{MMLU_BUDGET * 100:.1f} pt)",
             fontsize=13 * S, color=INK, pad=10)
ax.legend(loc="upper right", frameon=False, fontsize=10 * S, ncol=3)

fig.suptitle("Editing one component family to unlearn hazardous biology",
             fontsize=16 * S, color=INK, x=0.055, ha="left", y=0.975)
fig.text(0.055, 0.925,
         "WMDP-bio scored on 823 questions the edit never saw; the edit is fit on 300 others. "
         "Only the components' own weight mass is scaled —\n"
         "0.0024% of the model at k=4 — with one scalar per matrix per component. "
         "Unlike the German edit there is no free corner: past roughly a third of\n"
         "the removable knowledge, general capability starts coming out with it. "
         "Adding components buys a strictly better trade-off, not a clean one.",
         fontsize=9.8 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/wmdp_unlearn{ext}", facecolor="white", **kw)
print("wrote out/wmdp_unlearn.png/.pdf/.svg")
for label, f, _, _ in runs:
    p = pick(data[f])
    print(f"  {label}: bio {p['mcq']['bio_eval']['acc']:.3f} "
          f"(base {B_BIO:.3f}) mmlu {p['mcq']['mmlu']['acc']:.3f} "
          f"(base {B_MMLU:.3f}) lr={p['lr']:g} lam={p['lam']:g} "
          f"step={p['step']} bioRetainΔ={p['text']['bio_retain']['delta']:+.3f}")
