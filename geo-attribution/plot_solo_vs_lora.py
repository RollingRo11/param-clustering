"""Single-component parameter edit vs LoRA r=1, matched budget and objective.

Both arms: 2,048 German training tokens, objective
relu(log V - German CE) + 10*KL_en + 10*mean KL_romance, and the LoRA arm is
the best of its four (lambda, lr) configs under the same guarded selection rule
the component search used - i.e. the comparison is drawn in LoRA's favour.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
CHANCE = 9.096          # ln(128256) - base German CE 2.664

solo = json.loads((RUN / "german_solo.json").read_text())["best"]["detail"]
lora_rows = [r for r in json.loads((RUN / "budget_race_lora.json").read_text())
             if r["budget"] == 2048]
score = lambda r: (r["german"] - 10 * max(0.0, r["english"] - 0.15)
                   - 4 * max(0.0, r["romance"] - 0.35))
lora = max(lora_rows, key=score)

COMP, LORA = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = [("Single component (c3634)\n112 scalars", COMP),
          ("LoRA r=1\n704,512 parameters", LORA)]

german = [solo["german_europarl"], lora["german"]]
LANGS = [("English\n(Pile)", "english_pile", "english"),
         ("French", "french_europarl_heldout", "fr"),
         ("Spanish", "spanish_europarl_heldout", "es"),
         ("Italian", "italian_europarl_heldout", "it")]
collat = [[solo[k] for _, k, _ in LANGS],
          [lora["detail"][k] for _, _, k in LANGS]]

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(12.4, 5.6), facecolor="white",
    gridspec_kw={"width_ratios": [1.0, 1.55], "wspace": 0.26})
fig.subplots_adjust(top=0.735, bottom=0.12, left=0.055, right=0.985)


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=10.5)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9)


# ---- left: did it remove German? ----
chrome(axL)
xs = [0, 1]
axL.bar(xs, german, width=0.52, color=[COMP, LORA], zorder=3)
axL.axhline(CHANCE, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
axL.annotate("German at chance (+9.10)", xy=(-0.58, CHANCE - 0.55),
             fontsize=9.5, color=INK2, ha="left", va="top")
for x, v in zip(xs, german):
    axL.annotate(f"+{v:.2f}", xy=(x, v), xytext=(x, v + 0.6), ha="center",
                 fontsize=13, fontweight="600", color=INK)
axL.set_xticks(xs)
axL.set_xticklabels(["Single\ncomponent", "LoRA r=1"], fontsize=11, color=INK)
axL.set_xlim(-0.62, 1.62)
axL.set_ylim(0, 27.5)
axL.set_ylabel("German ΔCE (nats)", fontsize=11, color=INK)
axL.set_title("German removed — both past chance", fontsize=12.5,
              color=INK, pad=10, loc="left")

# ---- right: what else broke? ----
chrome(axR)
W = 0.36
base = list(range(len(LANGS)))
for i, ((label, color), vals) in enumerate(zip(SERIES, collat)):
    pos = [b + (i - 0.5) * W for b in base]
    axR.bar(pos, vals, width=W - 0.03, color=color, label=label, zorder=3)
    for x, v in zip(pos, vals):
        axR.annotate(f"{v:.3f}" if v < 0.1 else f"{v:.2f}", xy=(x, v),
                     xytext=(x, v + 0.055), ha="center", fontsize=10.5,
                     fontweight="600", color=color)
axR.set_xticks(base)
axR.set_xticklabels([n for n, _, _ in LANGS], fontsize=11, color=INK)
axR.set_ylim(0, 2.85)
axR.set_ylabel("Collateral ΔCE (nats) — lower is better", fontsize=11, color=INK)
axR.set_title("What else broke — 191× more English damage from LoRA",
              fontsize=12.5, color=INK, pad=10, loc="left")
axR.legend(frameon=False, fontsize=10.5, loc="upper right",
           labelspacing=0.9, handlelength=1.3)

fig.suptitle("Erasing German from Llama-3.2-1B: one parameter component vs a LoRA fine-tune",
             fontsize=15, color=INK, x=0.055, ha="left", y=0.965)
fig.text(0.055, 0.905,
         "Matched: 2,048 German training tokens, identical objective (German CE ↑ to "
         "chance, KL-preserve on English + French/Spanish/Italian).\n"
         "The LoRA arm is the best of its four configs under the same selection rule — "
         "the comparison is drawn in LoRA's favour.",
         fontsize=10, color=MUTED, ha="left", va="top", linespacing=1.5)

out = GEO / "out/solo_vs_lora.png"
fig.savefig(out, dpi=190, facecolor="white")
print("wrote", out)
print("component:", {k: round(solo[k], 4) for _, k, _ in LANGS},
      "de", round(solo["german_europarl"], 3))
print("lora:", lora["detail"], "lam", lora["lam_en"], "lr", lora["lr"])
print("english ratio: %.0fx" % (lora["detail"]["english"] / solo["english_pile"]))
