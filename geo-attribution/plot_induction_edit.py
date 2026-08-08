"""The induction dissociation, as one figure.

Scaling three components and watching two things at once:

  copy-2 cross-entropy   — can the model still copy?
  L10H23 induction score — do the heads still LOOK in the right place?

c3392 takes both down together: the mechanism is gone.
c108 takes the loss to 16 nats while the attention pattern never moves: the
heads still find the token, the model just can't say it.
c2747 — the component attribution ranked FIRST — barely moves either.
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
S = 1.5
GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
d = json.loads((RUN / "induction_demo.json").read_text())

SERIES = [("3392", "#2a78d6", "c3392  — the circuit", "o"),
          ("108",  "#eb6834", "c108  — the readout", "s"),
          ("2747", "#1baf7a", "c2747 — attribution's top pick", "^")]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.2, length=5)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.6), facecolor="white")
fig.subplots_adjust(top=0.575, bottom=0.135, left=0.055, right=0.995, wspace=0.24)

# --- panel 1: can it still copy? ---
ax = axes[0]
chrome(ax)
for key, color, lab, mk in SERIES:
    c = d["sweeps"][key]
    ax.plot([p["alpha"] for p in c], [p["copy2_ce"] for p in c], color=color,
            marker=mk, linewidth=2.4 * S, markersize=6 * S, label=lab,
            markeredgecolor="white", markeredgewidth=1.1 * S)
ax.axhline(d["base"]["copy2_ce"], color=MUTED, ls=(0, (4, 3)), lw=1.2 * S)
ax.annotate("unedited model", xy=(21.5, d["base"]["copy2_ce"] + 0.75),
            fontsize=8.8 * S, color=MUTED)
ax.set_xlabel("edit strength  α", fontsize=10.5 * S, color=INK)
ax.set_ylabel("copy-2 cross-entropy (nats)", fontsize=10.5 * S, color=INK)
ax.set_title("Can the model still copy?", fontsize=12.5 * S, color=INK, pad=10)

# --- panel 2: do the heads still look? ---
ax = axes[1]
chrome(ax)
for key, color, lab, mk in SERIES:
    c = d["sweeps"][key]
    ax.plot([p["alpha"] for p in c], [p["l10h23"] for p in c], color=color,
            marker=mk, linewidth=2.4 * S, markersize=6 * S,
            markeredgecolor="white", markeredgewidth=1.1 * S)
ax.set_ylim(0, 1.05)
ax.set_xlabel("edit strength  α", fontsize=10.5 * S, color=INK)
ax.set_ylabel("L10H23 induction score", fontsize=10.5 * S, color=INK)
ax.set_title("Do the induction heads still look in the right place?",
             fontsize=12.5 * S, color=INK, pad=10)

# --- panel 3: the dissociation, directly ---
ax = axes[2]
chrome(ax)
for key, color, lab, mk in SERIES:
    c = d["sweeps"][key]
    ax.plot([p["copy2_ce"] for p in c], [p["l10h23"] for p in c], color=color,
            marker=mk, linewidth=2.4 * S, markersize=6 * S,
            markeredgecolor="white", markeredgewidth=1.1 * S)
ax.set_ylim(0, 1.05)
ax.set_xlabel("copy-2 cross-entropy (nats)", fontsize=10.5 * S, color=INK)
ax.set_ylabel("L10H23 induction score", fontsize=10.5 * S, color=INK)
ax.set_title("Mechanism vs readout", fontsize=12.5 * S, color=INK, pad=10)
ax.annotate("heads intact,\nmodel cannot copy", xy=(12.0, 0.97),
            xytext=(6.2, 0.66), fontsize=9.2 * S, color="#eb6834",
            arrowprops=dict(arrowstyle="->", color="#eb6834", lw=1.6))
ax.annotate("heads destroyed", xy=(3.0, 0.20), xytext=(4.6, 0.30),
            fontsize=9.2 * S, color="#2a78d6",
            arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1.6))

handles = [plt.Line2D([], [], color=c, marker=mk, lw=2.4 * S,
                      markersize=6 * S, label=lab)
           for _, c, lab, mk in SERIES]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 0.715),
           ncol=3, frameon=False, fontsize=10.5 * S)
fig.suptitle("Editing induction: two components, two different failures",
             fontsize=15.5 * S, color=INK, x=0.055, ha="left", y=0.972)
fig.text(0.055, 0.915,
         "Each component's owned weight mass is scaled by α in all 112 matrices. "
         "The induction score is how much attention flows from a\nrepeated token to "
         "the token that followed it the first time — the mechanism itself, measured "
         "independently of the loss.",
         fontsize=9.8 * S, color=MUTED, ha="left", va="top", linespacing=1.55)

for ext, kw in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 400})):
    p = GEO / f"out/induction_edit{ext}"
    fig.savefig(p, facecolor="white", **kw)
    print("wrote", p.name)
