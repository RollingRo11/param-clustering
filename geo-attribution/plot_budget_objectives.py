"""Per-budget component-vs-LoRA figures, for both preserve objectives.

A: multilingual objective  — German removal and collateral at each budget.
B: english-only objective  — Romance is left undefended on purpose; the
   question is which method salts it.

Both arms: c3634 alone (112 per-matrix gains, invert init) vs LoRA r=1 on the
same 112 matrices, identical data, budget and step count (800). The LoRA arm
shown at each budget is whichever of its two learning rates best satisfies the
objective it was actually given.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
BUDGETS = [8, 64, 512, 2048]
CHANCE = 9.096
COMP, LORA = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

rows = json.loads((RUN / "budget_race_solo.json").read_text())


def score(r):
    s = r["german"] - 10 * max(0.0, r["english"] - 0.15)
    if r["objective"] == "multilingual":
        s -= 4 * max(0.0, r["romance"] - 0.35)
    return s


def pick(arm, obj, budget):
    cand = [r for r in rows if r["arm"] == arm and r["objective"] == obj
            and r["budget"] == budget]
    return max(cand, key=score)


def chrome(ax, ygrid=True):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.9)


LANGS = [("English", "english"), ("French", "fr"),
         ("Spanish", "es"), ("Italian", "it")]


def figure(obj, fname, title, sub, ycap=None):
    fig, axes = plt.subplots(2, len(BUDGETS), figsize=(13.6, 6.9),
                             facecolor="white", sharey="row")
    fig.subplots_adjust(top=0.715, bottom=0.115, left=0.062, right=0.99,
                        hspace=0.38, wspace=0.12)
    for col, B in enumerate(BUDGETS):
        c, l = pick("component", obj, B), pick("lora", obj, B)
        ax = axes[0][col]
        chrome(ax)
        ax.bar([0, 1], [c["german"], l["german"]], width=0.55,
               color=[COMP, LORA], zorder=3)
        ax.axhline(CHANCE, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.1)
        for x, v in zip([0, 1], [c["german"], l["german"]]):
            ax.annotate(f"{v:.1f}", xy=(x, v), xytext=(x, v + 0.45),
                        ha="center", fontsize=10.5, fontweight="600", color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["component", "LoRA"], fontsize=9.5, color=INK)
        ax.set_xlim(-0.62, 1.62)
        ax.set_ylim(0, 13.6)
        ax.set_title(f"{B:,} German tokens", fontsize=11.5, color=INK, pad=8)
        if col == 0:
            ax.set_ylabel("German removed\nΔCE (nats)", fontsize=10, color=INK)
            ax.annotate("chance", xy=(-0.55, CHANCE + 0.25), fontsize=8.5,
                        color=MUTED)

        ax = axes[1][col]
        chrome(ax)
        base = np.arange(len(LANGS))
        W = 0.38
        top = max(max(r["detail"][k] for _, k in LANGS) for r in (c, l))
        for i, (row, color) in enumerate(((c, COMP), (l, LORA))):
            vals = [row["detail"][k] for _, k in LANGS]
            ax.bar(base + (i - 0.5) * W, vals, width=W - 0.04, color=color,
                   zorder=3)
            # every bar carries its number: on a linear axis the component
            # bars are invisibly small, which is the finding, not a defect
            for x, v in zip(base + (i - 0.5) * W, vals):
                ax.annotate(f"{v:.2f}" if abs(v) >= 0.1 else f"{v:.3f}",
                            xy=(x, max(v, 0)),
                            xytext=(x, max(v, 0) + top * 0.035),
                            ha="center", fontsize=8, fontweight="600",
                            color=color, rotation=90)
        ax.set_ylim(0, top * 1.30)
        ax.set_xticks(base)
        ax.set_xticklabels([n for n, _ in LANGS], fontsize=9, color=INK,
                           rotation=30, ha="right")
        if col == 0:
            ax.set_ylabel("collateral ΔCE (nats)", fontsize=10, color=INK)
        worst_c = max(c["detail"][k] for _, k in LANGS)
        worst_l = max(l["detail"][k] for _, k in LANGS)
        ax.set_title(f"worst: {worst_c:.2f} vs {worst_l:.2f} nats",
                     fontsize=9.5, color=MUTED, pad=6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COMP),
               plt.Rectangle((0, 0), 1, 1, color=LORA),
               plt.Line2D([], [], linestyle=(0, (4, 3)), color=MUTED)]
    fig.legend(handles, ["single component c3634 (112 scalars)",
                         "LoRA r=1 (704,512 parameters)",
                         "German at chance (+9.10)"],
               loc="upper left", bbox_to_anchor=(0.060, 0.815), ncol=3,
               frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=15, color=INK, x=0.062, ha="left", y=0.965)
    fig.text(0.062, 0.905, sub, fontsize=10, color=MUTED, ha="left",
             va="top", linespacing=1.5)
    out = GEO / fname
    fig.savefig(out, dpi=170, facecolor="white")
    print("wrote", out)
    for B in BUDGETS:
        c, l = pick("component", obj, B), pick("lora", obj, B)
        print(f"  B={B:<5} component de={c['german']:+6.2f} "
              f"en={c['english']:+6.2f} rom={c['romance']:+6.2f}   |   "
              f"lora(lr={l['lr']:g}) de={l['german']:+6.2f} "
              f"en={l['english']:+6.2f} rom={l['romance']:+6.2f}")


figure("multilingual", "out/budget_lora_vs_component.png",
       "German removal vs collateral, by training-token budget — one component vs LoRA r=1",
       "Multilingual objective: German CE ↑ to chance, KL-preserve on English AND French/Spanish/Italian.\n"
       "Identical data, budget and 800 steps for both arms; the LoRA learning rate shown is whichever better satisfies the objective.",
       20)
print()
figure("english_only", "out/english_only_collateral.png",
       "Protect English only — who salts the other languages?",
       "Same setup, but the Romance KL term is REMOVED: nothing defends French, Spanish or Italian.\n"
       "The component edit stays selective without being told to; LoRA takes the shortest path, which is 'destroy everything that isn't English'.",
       20)
