"""Collateral at MATCHED German removal — the targeted-ness question, directly.

For each method we have many (lr, lambda, step) points. Rather than pick one
configuration each — which is what made the earlier figure unfair to LoRA —
we ask, for every German-removal target G:

    what is the least collateral damage this method can do
    while still removing at least G nats of German?

Points are CHOSEN on dev/guard blocks and REPORTED on held-out blocks, so a
method with more hyperparameters to tune cannot win by overfitting the
selection.

A method being "more targeted" means its curve sits lower: less damage for the
same removal. If the curves cross, the honest statement is that neither
dominates and it depends on how much German you want gone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Poster output: vector PDF is the deliverable (infinite resolution at any
# print size); the PNG is a 400-dpi convenience copy. fonttype 42 embeds
# TrueType rather than Type-3, which print shops and Illustrator both prefer.
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.linewidth": 1.2,
    "figure.dpi": 120,
})
SCALE = 1.55          # type/line scale for viewing at poster distance


def save(fig, stem):
    """Write vector PDF + SVG + high-dpi PNG for the same figure."""
    outs = []
    for ext, kw in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 400})):
        p = GEO / f"{stem}{ext}"
        fig.savefig(p, facecolor="white", **kw)
        outs.append(p.name)
    print("wrote " + ", ".join(outs))

GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
BUDGETS = [8, 64, 512, 2048]
CHANCE = 9.096
COMP, LORA = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
EN_BUDGET = 0.15          # dev English leak both methods must respect

rows = json.loads((RUN / "lora_fair_sweep.json").read_text())


def best_at(points, target, key):
    """Least eval `key` among points whose DEV German removal >= target."""
    ok = [p for p in points
          if p["dev"]["german"] >= target
          and p["dev"]["english"] <= EN_BUDGET]
    if not ok:
        return None
    return min(ok, key=lambda p: p["dev"][key])["eval"][key]


def figure(obj, key, fname, title, sub, ylab):
    fig, axes = plt.subplots(1, len(BUDGETS), figsize=(20.0, 6.2),
                             facecolor="white", sharey=True)
    fig.subplots_adjust(top=0.615, bottom=0.145, left=0.058, right=0.995,
                        wspace=0.09)
    summary = []
    for col, B in enumerate(BUDGETS):
        ax = axes[col]
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=9.5*SCALE, width=1.2, length=5)
        ax.set_axisbelow(True)
        ax.grid(True, color=GRID, linewidth=0.9*SCALE)
        pts = {a: [p for p in rows if p["objective"] == obj
                   and p["budget"] == B and p["arm"] == a]
               for a in ("component", "lora")}
        hi = max((p["dev"]["german"] for v in pts.values() for p in v),
                 default=1.0)
        targets = np.linspace(0.5, min(hi, 26), 60)
        for arm, color, lab in ((("component"), COMP, "single component"),
                                (("lora"), LORA, "LoRA r=1")):
            xs, ys = [], []
            for t in targets:
                v = best_at(pts[arm], t, key)
                if v is not None:
                    xs.append(t)
                    ys.append(v)
            if xs:
                ax.plot(xs, ys, color=color, linewidth=2.2*SCALE, label=lab,
                        solid_capstyle="round", zorder=4)
                ax.plot([xs[-1]], [ys[-1]], marker="o", color=color,
                        markersize=6*SCALE, markeredgecolor="white",
                        markeredgewidth=1.2*SCALE, zorder=5)
        ax.axvline(CHANCE, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1*SCALE)
        ax.set_xlabel("German removed, ΔCE (nats)", fontsize=10*SCALE, color=INK)
        ax.set_title(f"{B:,} German tokens", fontsize=11.5*SCALE, color=INK, pad=10)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=10*SCALE, color=INK)
            ax.annotate("German\nat chance", xy=(CHANCE - 0.35, ax.get_ylim()[1] * 0.90),
                        fontsize=8.5*SCALE, color=MUTED, ha="right", va="top")
        # matched comparison at the component's own reach
        cmax = max((p["dev"]["german"] for p in pts["component"]
                    if p["dev"]["english"] <= EN_BUDGET), default=None)
        if cmax:
            c = best_at(pts["component"], cmax, key)
            l = best_at(pts["lora"], cmax, key)
            summary.append((B, cmax, c, l))
    handles = [plt.Line2D([], [], color=COMP, lw=2.2*SCALE, label="single component c3634 (112 scalars)"),
               plt.Line2D([], [], color=LORA, lw=2.2*SCALE, label="LoRA r=1 (704,512 params)"),
               plt.Line2D([], [], color=MUTED, ls=(0, (4, 3)), lw=1.4*SCALE, label="German at chance")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.735),
               ncol=3, frameon=False, fontsize=10*SCALE)
    fig.suptitle(title, fontsize=15*SCALE, color=INK, x=0.058, ha="left", y=0.972)
    fig.text(0.058, 0.915, sub, fontsize=9.8*SCALE, color=MUTED, ha="left",
             va="top", linespacing=1.55)
    save(fig, fname)
    print(f"  matched at the component's own maximum German removal:")
    for B, g, c, l in summary:
        if c is None or l is None:
            continue
        ratio = f"{l / c:.1f}x" if c > 1e-6 else "n/a"
        print(f"    B={B:<5} at de>={g:5.2f}:  component {c:6.3f}   "
              f"lora {l:6.3f}   ({ratio})")


figure("english_only", "romance", "out/fair_frontier_english_only",
       "Collateral at matched German removal — English protected, Romance left undefended",
       "Each curve: the least Romance damage that method can achieve while removing at least X nats of German, over a sweep of\n"
       "(learning rate, lambda, training step). Points chosen on dev/guard blocks, reported on held-out blocks. Lower = more targeted.",
       "worst Romance ΔCE (nats)")
print()
figure("multilingual", "romance", "out/fair_frontier_multilingual",
       "Collateral at matched German removal — English and Romance both protected",
       "Same sweep and selection, with the Romance KL term active for both methods.",
       "worst Romance ΔCE (nats)")
print()
figure("english_only", "english", "out/fair_frontier_english_leak",
       "English leak at matched German removal — English protected for both",
       "The quantity both objectives explicitly defend; a sanity check that neither method is winning by ignoring it.",
       "English ΔCE (nats)")
