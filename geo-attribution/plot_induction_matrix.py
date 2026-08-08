"""Where in the model does the induction-ranked component set route induction?

Each of the 112 per-matrix scalars is perturbed alone (the other 111 stay at
identity), so a hot cell means that matrix-slice of those components carries
induction. Damage to ordinary text is shown alongside, because the last layer
moves logits for everything and would otherwise read as a false positive.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

GEO = Path("/workspace/param-clustering/geo-attribution")
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
ALPHA = 16.0
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
         "down_proj"]
SHORT = ["q", "k", "v", "o", "gate", "up", "down"]

d = json.loads((RUN / "induction_matrix.json").read_text())
rows = [r for r in d["rows"] if r["alpha"] == ALPHA]
L = max(r["layer"] for r in rows) + 1
ind = np.zeros((len(KINDS), L))
ctl = np.zeros((len(KINDS), L))
for r in rows:
    ind[KINDS.index(r["kind"]), r["layer"]] = r["d_induction"]
    ctl[KINDS.index(r["kind"]), r["layer"]] = r["d_control"]

# documented sequential blue ramp, light -> dark
BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
        "#184f95", "#0d366b"]
cmap = LinearSegmentedColormap.from_list("seqblue", BLUE)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"

fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.0), facecolor="white")
fig.subplots_adjust(top=0.745, bottom=0.085, left=0.075, right=0.955,
                    hspace=0.40)

for ax, mat, title, unit in (
        (axes[0], ind,
         "Induction damage — ΔCE on the repeated copy (base 0.095 nats)", "nats"),
        (axes[1], ctl,
         "Collateral — ΔCE on ordinary English (base 3.045 nats)", "nats")):
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0,
                   vmax=max(mat.max(), 1e-9), interpolation="nearest")
    ax.set_xticks(range(L))
    ax.set_xticklabels(range(L), fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(SHORT)))
    ax.set_yticklabels(SHORT, fontsize=10, color=INK2)
    ax.set_xlabel("layer", fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=11.5, color=INK, pad=8, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SHORT), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.030)
    cb.ax.tick_params(labelsize=9, colors=INK2, length=0)
    cb.outline.set_visible(False)
    cb.set_label(unit, fontsize=9.5, color=INK2)
    # label the hottest cells
    flat = np.dstack(np.unravel_index(np.argsort(-mat, axis=None), mat.shape))[0]
    for r, c in flat[:4]:
        if mat[r, c] <= 0:
            continue
        ax.annotate(f"{mat[r, c]:.3f}", xy=(c, r), ha="center", va="center",
                    fontsize=8.5, fontweight="600",
                    color="white" if mat[r, c] > 0.55 * mat.max() else INK)

fig.suptitle("Where induction lives: sweeping the 112 per-matrix scalars one at a time",
             fontsize=15, color=INK, x=0.075, ha="left", y=0.965)
fig.text(0.075, 0.912,
         f"Components {d['components']} scaled by α={ALPHA:g} in ONE matrix at a time;\n"
         "the other 111 stay at identity. Induction concentrates in the layer 8–9 MLP — 98% of all "
         "damage is MLP, none of it attention.\nThe final layer's apparent effect is the generic "
         "'last-layer moves logits' confound: it is the only warm region in the collateral panel.",
         fontsize=10, color=MUTED, ha="left", va="top", linespacing=1.5)

out = GEO / "out/induction_matrix.png"
fig.savefig(out, dpi=175, facecolor="white")
print("wrote", out)
order = sorted(rows, key=lambda r: -r["d_induction"])[:8]
for r in order:
    sel = r["d_induction"] / max(r["d_control"], 1e-4)
    print(f"  L{r['layer']:<2} {r['kind']:<10} induction {r['d_induction']:+7.3f} "
          f"control {r['d_control']:+7.4f}  sel {sel:8.1f}x")
by_layer = ind.sum(0)
print("\nlayer totals (induction ΔCE):",
      ", ".join(f"L{i}:{v:.3f}" for i, v in enumerate(by_layer) if v > 0.01))
print("MLP share of total induction damage: %.0f%%"
      % (100 * ind[4:].sum() / ind.sum()))
