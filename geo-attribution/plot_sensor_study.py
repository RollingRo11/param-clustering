"""Ablatability curves by attribution sensor, on VPD's 67M Pile target.

Nine sensors is past the point where overlapping lines stay readable, so the
curves are faceted by family -- EAP/IG on the left, GIM variants and the two
floors in the middle -- with GIM repeated as a dashed reference in both so the
panels can be compared directly. The bar panel carries all nine.

EAP is IG with K=1, so eap/ig2/ig3/ig5 is an ORDERED variable and gets a
single-hue light-to-dark ramp (validated as an ordinal ramp); the GIM variants
and floors are categorical (validated all-pairs).

    python3.12 plot_sensor_study.py --glob 'out/sensors/c256_s*.json'
"""
import argparse
import glob
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
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# key, colour, marker, legend label, bar label
IG = [
    ("eap", "#6bb0dd", "o", "EAP  (grad × act) = IG, K=1", "EAP\nK=1"),
    ("ig2", "#2f86c4", "o", "IG, K=2", "IG\nK=2"),
    ("ig3", "#0f5ca3", "o", "IG, K=3", "IG\nK=3"),
    ("ig5", "#083a72", "o", "IG, K=5", "IG\nK=5"),
]
GIMS = [
    ("gim", "#0b0b0b", "s", "GIM  (production sensor)", "GIM\nfull"),
    ("gim_softmax_only", "#e8590c", "^", "GIM: τ-softmax only", "GIM\nτ only"),
    ("gim_scales_only", "#9c36b5", "v", "GIM: other 3 mods only", "GIM\n3 mods"),
    ("actonly", "#6f6d66", "D", "|activation| only (no gradient)", "act\nonly"),
    ("random", "#9d9b90", "D", "random  (null floor)", "random"),
]
ALL = IG + GIMS

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default=str(GEO / "out/sensors/c256_s*.json"))
ap.add_argument("--out", type=Path, default=GEO / "out/sensor_study")
ap.add_argument("--title", default="Which attribution sensor yields the most "
                                   "ablatable decomposition?")
ap.add_argument("--note", default="")
args = ap.parse_args()

files = sorted(glob.glob(args.glob))
ds = [json.loads(Path(f).read_text()) for f in files]
assert ds, f"no files matched {args.glob}"
base, C = ds[0]["base_ce"], ds[0]["C"]
ks = np.array([c["k"] for c in ds[0]["results"]["gim"]["curve"]], float)
pos = ks > 0


def band(key):
    y = np.array([[d["results"][key]["curve"][i]["ce"] - base
                   for i in range(len(ks))] for d in ds])
    return (np.maximum(y.mean(0), 1e-6), np.maximum(y.min(0), 1e-6),
            np.maximum(y.max(0), 1e-6))


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 3, figsize=(23.0, 7.4), facecolor="white",
                         gridspec_kw={"width_ratios": [1, 1, 1.28]})
fig.subplots_adjust(top=0.585, bottom=0.155, left=0.046, right=0.991,
                    wspace=0.215)

YLO, YHI = 6e-4, 45
for ax, group, ttl in ((axes[0], IG, "EAP and integrated gradients"),
                       (axes[1], GIMS, "GIM variants, and the floors")):
    chrome(ax)
    if group is IG:                      # GIM as a dashed reference line
        mu, _, _ = band("gim")
        ax.plot(ks[pos], mu[pos], color=INK, ls=(0, (4, 3)), lw=2.0 * S,
                label="GIM  (reference)", zorder=2)
    for key, col, mk, lab, _ in group:
        mu, lo, hi = band(key)
        ax.plot(ks[pos], mu[pos], color=col, marker=mk, lw=2.5 * S,
                markersize=5.8 * S, markeredgecolor="white",
                markeredgewidth=1.0 * S, label=lab, zorder=3)
        ax.fill_between(ks[pos], lo[pos], hi[pos], color=col, alpha=0.14,
                        linewidth=0, zorder=1)
    ax.axhline(0.05, color=INK, ls=(0, (1, 2)), lw=1.5 * S, zorder=2)
    ax.annotate("ΔCE = 0.05", xy=(C * 0.95, 0.058), ha="right",
                fontsize=9.6 * S, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3.5, C * 1.05)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel(f"components deleted, lightest first  (of {C})",
                  fontsize=10.5 * S, color=INK)
    ax.set_ylabel("ΔCE from the unedited model (nats)", fontsize=10.5 * S,
                  color=INK)
    ax.set_title(ttl, fontsize=12.5 * S, color=INK, pad=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9.3 * S,
              handlelength=2.1, borderpad=0.1, labelspacing=0.35)

# ---- the headline number ----
ax = axes[2]
chrome(ax)
rng = np.random.default_rng(0)
vals = {}
for i, (key, col, _, _, short) in enumerate(ALL):
    v = np.array([d["results"][key]["removable_within_0.05"] for d in ds],
                 dtype=float)
    vals[key] = v
    ax.bar(i, v.mean(), 0.64, color=col, zorder=3)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.115, 0.115, len(v)), v,
               s=76 * S, color="white", edgecolors=INK, linewidths=1.35 * S,
               zorder=5)
    ax.annotate(f"{v.mean():.0f}", (i, v.max()), (i, v.max() + 1.5),
                ha="center", fontsize=11.5 * S, fontweight="600", color=INK)
ax.set_xticks(range(len(ALL)))
ax.set_xticklabels([s[4] for s in ALL], fontsize=8.6 * S, color=INK,
                   linespacing=1.4)
ax.set_xlim(-0.62, len(ALL) - 0.38)
ax.set_ylim(0, max(v.max() for v in vals.values()) * 1.24)
ax.axvline(3.5, color=AXIS, lw=1.1 * S, ls=(0, (3, 3)), zorder=1)
ax.axvline(6.5, color=AXIS, lw=1.1 * S, ls=(0, (3, 3)), zorder=1)
ax.set_ylabel(f"components removable within ΔCE 0.05  (of {C})",
              fontsize=10.5 * S, color=INK)
ax.set_title("Higher = more of the model is inert under this sensor",
             fontsize=12.5 * S, color=INK, pad=11)
ax.annotate(f"white dots = the {len(ds)} seeds", xy=(0.975, 0.955),
            xycoords="axes fraction", ha="right", fontsize=9.4 * S, color=INK2)

fig.suptitle(args.title, fontsize=15.5 * S, color=INK, x=0.046, ha="left",
             y=0.972)
fig.text(0.046, 0.905, args.note, fontsize=9.6 * S, color=MUTED, ha="left",
         va="top", linespacing=1.62)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(f"{args.out}{ext}", facecolor="white", **kw)
print(f"wrote {args.out}.png/.pdf/.svg   base={base:.4f} C={C} seeds={len(ds)}")
for key, *_ in ALL:
    v = vals[key]
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    print(f"  {key:<18} mean {v.mean():5.1f} +- {sd:4.1f}")
