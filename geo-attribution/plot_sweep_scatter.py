"""German removal vs per-language damage, every point in the sweep.

The bar figures show one chosen operating point per method. This shows the
whole cloud: each marker is one (learning rate, lambda, training step) setting,
so the shape of each method's reachable trade-off is visible rather than
summarised. Rows are languages, columns are German token budgets.

Points off the top of a panel are drawn as open markers on the ceiling, so the
axis stays linear and readable without hiding the configurations that blew up.
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
    "axes.linewidth": 1.1,
})
S = 1.35
GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
BUDGETS = [8, 64, 512, 2048]
LANGS = [("English (Pile)", "english"), ("French", "fr"),
         ("Spanish", "es"), ("Italian", "it")]
CHANCE = 9.096
COMP, LORA = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
YCAP = {"english": 1.0, "fr": 1.4, "es": 1.4, "it": 1.4}
XCAP = 15.0          # nats past chance; beyond this the exact overshoot is moot

rows = json.loads((RUN / "lora_fair_sweep.json").read_text())


def figure(obj, fname, title, sub):
    fig, axes = plt.subplots(len(LANGS), len(BUDGETS), figsize=(18.0, 15.0),
                             facecolor="white")
    fig.subplots_adjust(top=0.845, bottom=0.055, left=0.068, right=0.995,
                        hspace=0.30, wspace=0.13)
    for r, (lname, lkey) in enumerate(LANGS):
        cap = YCAP[lkey]
        for c, B in enumerate(BUDGETS):
            ax = axes[r][c]
            ax.set_facecolor("white")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(AXIS)
            ax.tick_params(colors=INK2, labelsize=9 * S, width=1.1, length=4)
            ax.set_axisbelow(True)
            ax.grid(True, color=GRID, linewidth=0.85 * S)
            ax.axhline(0, color=AXIS, linewidth=1.0 * S)
            ax.axvline(0, color=MUTED, linestyle=(0, (4, 3)),
                       linewidth=1.1 * S)
            for arm, color, mk in (("component", COMP, "o"),
                                   ("lora", LORA, "s")):
                pts = [p for p in rows if p["objective"] == obj
                       and p["budget"] == B and p["arm"] == arm]
                # x is measured against chance, not against the base model:
                # 0 means German has been driven all the way to uniform, and
                # the sign says whether a setting fell short or overshot.
                x = np.array([p["eval"]["german"] for p in pts]) - CHANCE
                y = np.array([p["eval"]["detail"][lkey] for p in pts])
                on = (y <= cap) & (x <= XCAP)
                ax.scatter(x[on], y[on], s=52 * S, marker=mk,
                           facecolors=color, edgecolors="white",
                           linewidths=0.9 * S, alpha=0.85, zorder=3)
                if (~on).any():
                    # off-scale in either direction: clamp to the edge and
                    # draw hollow. The value is not worth the clutter, but
                    # the configuration still existed and should be counted.
                    ax.scatter(np.minimum(x[~on], XCAP),
                               np.minimum(y[~on], cap), s=52 * S,
                               marker=mk, facecolors="white",
                               edgecolors=color, linewidths=1.5 * S, zorder=3)
            ax.set_xlim(-CHANCE - 1.0, XCAP + 0.9)
            ax.set_ylim(-cap * 0.10, cap * 1.06)
            if r == 0:
                ax.set_title(f"{B:,} German tokens", fontsize=12 * S,
                             color=INK, pad=10)
            if c == 0:
                ax.set_ylabel(f"{lname}\nΔCE (nats)", fontsize=10.5 * S,
                              color=INK)
            if r == len(LANGS) - 1:
                ax.set_xlabel("German CE relative to chance (nats)",
                              fontsize=10 * S, color=INK)
            if r == 0 and c == 0:
                ax.annotate("chance", xy=(-0.7, cap * 0.97),
                            fontsize=8.4 * S, color=MUTED, ha="right",
                            va="top")
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=9 * S,
                          markerfacecolor=COMP, markeredgecolor="white",
                          label="single component c3634 (112 scalars)"),
               plt.Line2D([], [], linestyle="", marker="s", markersize=9 * S,
                          markerfacecolor=LORA, markeredgecolor="white",
                          label="LoRA r=1 (704,512 parameters)"),
               plt.Line2D([], [], linestyle="", marker="o", markersize=9 * S,
                          markerfacecolor="white", markeredgecolor=MUTED,
                          label="open = off-scale, clamped to the axis edge")]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(0.068, 0.905), ncol=3, frameon=False,
               fontsize=10.5 * S)
    fig.suptitle(title, fontsize=16 * S, color=INK, x=0.068, ha="left",
                 y=0.978)
    fig.text(0.068, 0.955, sub, fontsize=10 * S, color=MUTED, ha="left",
             va="top", linespacing=1.5)
    for ext, kw in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
        fig.savefig(GEO / f"{fname}{ext}", facecolor="white", **kw)
    print(f"wrote {fname}.png/.pdf/.svg")
    for B in BUDGETS:
        for arm in ("component", "lora"):
            n = len([p for p in rows if p["objective"] == obj
                     and p["budget"] == B and p["arm"] == arm])
            print(f"  B={B:<5} {arm:<10} {n} swept points")


figure("english_only", "out/sweep_scatter_english_only",
       "Every configuration in the sweep — German removal vs collateral, English-only objective",
       "One marker per (learning rate, lambda, training step). x = 0 is German at chance: left of it the language partly survives, right of it\n"
       "the edit has pushed past chance. Only English is defended; French, Spanish and Italian are undefended, so their panels show what\n"
       "each method does to them incidentally. Held-out blocks throughout.")
