"""VPD Section 5 attribution graphs, drawn.

Nodes are components, placed left-to-right by DEPTH (mass-weighted mean layer
of their owned parameters). Node size is the direct effect on log p(target).
Arrows are causal edges: how much of the downstream node's effect disappears
once the upstream node is perturbed.

Two cases side by side because they come out completely differently, and that
contrast is the finding: the copy task is a star with one component doing
everything, while pronoun agreement is a distributed circuit.
"""
import json
import textwrap
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
S = 1.42
GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
POS, NEG = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

CASES = [("pronoun", "Pronoun agreement — a distributed circuit"),
         ("induction", "Copying a novel string — a star")]
graphs = {k: json.loads((RUN / f"attribution_graph_{k}.json").read_text())
          for k, _ in CASES}

fig, axes = plt.subplots(2, 1, figsize=(17.0, 10.4), facecolor="white")
fig.subplots_adjust(top=0.775, bottom=0.075, left=0.045, right=0.985,
                    hspace=0.30)

for ax, (key, title) in zip(axes, CASES):
    g = graphs[key]
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # one row, ordered by depth: arcs above the line carry the edges, so no
    # two labels ever compete for the same space
    nodes = sorted(g["nodes"], key=lambda n: n["depth"])
    de = np.array([n["direct_effect"] for n in nodes])
    scale = np.abs(de).max()
    x = np.arange(len(nodes), dtype=float)
    idx = {n["component"]: i for i, n in enumerate(nodes)}

    emax = max(abs(e["weight"]) for e in g["edges"]) if g["edges"] else 1.0
    for e in sorted(g["edges"], key=lambda e: abs(e["weight"]))[-12:]:
        a, b = idx[e["from_component"]], idx[e["to_component"]]
        w = e["weight"]
        lw = 0.7 + 5.0 * abs(w) / emax
        rad = 0.11 + 0.028 * abs(b - a)
        ax.annotate("", xy=(x[b], 0.10), xytext=(x[a], 0.10),
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=17 * S,
                                    color=POS if w > 0 else NEG, lw=lw * S,
                                    alpha=0.6, shrinkA=17, shrinkB=19,
                                    connectionstyle=f"arc3,rad={-rad}"))

    for i, n in enumerate(nodes):
        r = 780 + 2350 * abs(n["direct_effect"]) / scale
        col = POS if n["direct_effect"] > 0 else NEG
        ax.scatter(x[i], 0.0, s=r * S, color=col, zorder=4,
                   edgecolors="white", linewidths=2.2 * S)
        ax.annotate(f"c{n['component']}", (x[i], 0.0), ha="center",
                    va="center", fontsize=7.7 * S, color="white",
                    fontweight="600", zorder=6)
        ax.annotate(f"{n['direct_effect']:+.3f}", (x[i], 0.0), (0, -26),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=9.4 * S, color=col, fontweight="600", zorder=6)
        ax.annotate(textwrap.fill(n["label"], 24), (x[i], 0.0), (0, -44),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7.9 * S, color=INK2, linespacing=1.35, zorder=6)
        ax.annotate(f"L{n['depth']:.1f}", (x[i], 0.0), (0, 24),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.4 * S, color=MUTED, zorder=6)

    ax.set_xlim(-0.75, len(nodes) - 0.25)
    ax.set_ylim(-1.42, 0.55)
    ax.annotate(title, xy=(0.0, 1.10), xycoords="axes fraction", ha="left",
                fontsize=12.5 * S, color=INK, fontweight="600")
    ax.annotate(f"{g['prompt']!r}  →  {g['target']!r}",
                xy=(0.0, 1.035), xycoords="axes fraction", ha="left",
                fontsize=9.2 * S, color=INK2, family="monospace")
    ax.annotate("shallow  →  deep", xy=(1.0, 1.035),
                xycoords="axes fraction", ha="right",
                fontsize=9.2 * S, color=MUTED)

handles = [
    plt.Line2D([], [], marker="o", linestyle="", markersize=11 * S,
               markerfacecolor=POS, markeredgecolor="white",
               label="scaling HURTS the target"),
    plt.Line2D([], [], marker="o", linestyle="", markersize=11 * S,
               markerfacecolor=NEG, markeredgecolor="white",
               label="scaling HELPS the target"),
    plt.Line2D([], [], color=POS, lw=3.2 * S,
               label="arc: upstream carries downstream's effect"),
]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.045, 0.008),
           ncol=3, frameon=False, fontsize=10 * S)

fig.suptitle("Attribution graphs over parameter components — VPD §5",
             fontsize=15.5 * S, color=INK, x=0.045, ha="left", y=0.978)
fig.text(0.045, 0.947,
         "Node size is the direct effect on log p(target) when that component's owned mass is scaled by 4. "
         "An arc c′→c is DE(c) − DE(c | c′ already perturbed):\n"
         "how much of c's effect depends on c′ having run first. Order is by depth, the mass-weighted mean "
         "layer of a component's parameters. VPD ABLATES here; that\n"
         "fails on this decomposition — a geometric partition of trained weights, not something optimised "
         "for ablation sufficiency. Zeroing a whole component moves\n"
         "log p by ~0.002 nats; scaling it by 4 moves it by up to 5.",
         fontsize=9.5 * S, color=MUTED, ha="left", va="top", linespacing=1.62)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/attribution_graph{ext}", facecolor="white", **kw)
print("wrote out/attribution_graph.png/.pdf/.svg")
for k, _ in CASES:
    g = graphs[k]
    top = max(g["nodes"], key=lambda n: abs(n["direct_effect"]))
    share = abs(top["direct_effect"]) / sum(abs(n["direct_effect"])
                                            for n in g["nodes"])
    print(f"  {k:<10} top node c{top['component']} holds "
          f"{100 * share:.0f}% of total |DE|")
