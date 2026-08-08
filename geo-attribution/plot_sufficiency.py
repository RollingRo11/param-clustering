"""Per-input sufficiency curves: how few components compute one output token?

x = components KEPT (top-k by that input's attribution), y = cross-entropy of
that single target token. Every curve must return to the unablated CE at k=C,
because the softpart shares sum to 1 at every weight entry.

    python3.12 plot_sufficiency.py --glob 'out/suff/s*.json'
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

MIB = [
    ("eap_linmlp_gimattn", "#0f7f6c", "P", "EAP + lin-MLP + GIM attn   (MIB #1)"),
    ("gim", "#0b0b0b", "s", "GIM   (MIB #2, production sensor)"),
    ("eapig_inputs5", "#c026a0", "X", "EAP-IG-inputs, K=5   (MIB #3)"),
    ("eap_linmlp", "#e8590c", "^", "EAP + lin-MLP   (MIB #4)"),
]
OURS = [
    ("eap", "#6bb0dd", "o", "EAP  (grad × act)"),
    ("ig5", "#083a72", "o", "IG, K=5  (weight path)"),
    ("gim_softmax_only", "#9c36b5", "v", "GIM: τ-softmax only"),
    ("random", "#9d9b90", "D", "random decomposition"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default=str(GEO / "out/suff/s*.json"))
ap.add_argument("--out", type=Path, default=GEO / "out/sufficiency")
ap.add_argument("--title", default="How few components suffice to compute one "
                                   "output token?")
ap.add_argument("--note", default="")
ap.add_argument("--kbar", type=int, default=16)
ap.add_argument("--tol", type=float, default=0.25)
args = ap.parse_args()

ds = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(args.glob))]
assert ds, f"no files matched {args.glob}"
base, C = ds[0]["base_token_ce"], ds[0]["C"]
KEEP = np.array(ds[0]["keep_grid"], float)
pos = KEEP > 0
present = [g for g in MIB + OURS if g[0] in ds[0]["results"]]


def band(key, field="ce_attr"):
    y = np.array([d["results"][key][field] for d in ds])
    return y.mean(0), y.min(0), y.max(0)


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
                         gridspec_kw={"width_ratios": [1, 1, 1.2]})
fig.subplots_adjust(top=0.585, bottom=0.155, left=0.058, right=0.99, wspace=0.235)

for ax, group, ttl in (
        (axes[0], [g for g in MIB if g in present],
         "MIB leaderboard methods"),
        (axes[1], [g for g in OURS if g in present],
         "The EAP/IG family, and the floor")):
    chrome(ax)
    if group and group[0][0] != "gim" and "gim" in ds[0]["results"]:
        mu, _, _ = band("gim")
        ax.plot(KEEP[pos], mu[pos], color=INK, ls=(0, (4, 3)), lw=2.0 * S,
                label="GIM  (reference)", zorder=2)
    for key, col, mk, lab in group:
        mu, lo, hi = band(key)
        ax.plot(KEEP[pos], mu[pos], color=col, marker=mk, lw=2.5 * S,
                markersize=6.0 * S, markeredgecolor="white",
                markeredgewidth=1.0 * S, label=lab, zorder=3)
        ax.fill_between(KEEP[pos], lo[pos], hi[pos], color=col, alpha=0.14,
                        linewidth=0, zorder=1)
    mu_r, _, _ = band(present[0][0], "ce_random")
    ax.plot(KEEP[pos], mu_r[pos], color=MUTED, ls=(0, (1, 2)), lw=2.0 * S,
            label="random ranking of the same components", zorder=2)
    ax.axhline(base, color=INK, ls=(0, (6, 3)), lw=1.5 * S, zorder=2)
    ax.annotate(f"unablated, {base:.2f}", xy=(C * 0.97, base * 1.16),
                ha="right", fontsize=9.6 * S, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, C * 1.1)
    ax.set_xlabel(f"components KEPT, highest attribution first  (of {C})",
                  fontsize=10.5 * S, color=INK)
    ax.set_ylabel("cross-entropy of the target token (nats)",
                  fontsize=10.5 * S, color=INK)
    ax.set_title(ttl, fontsize=12.5 * S, color=INK, pad=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9.2 * S,
              handlelength=2.1, borderpad=0.1, labelspacing=0.34)

# ---- bar: smallest budget that reproduces the token ----
ax = axes[2]
chrome(ax)
KL = list(ds[0]["keep_grid"])


def thresh(key, d):
    m = d["results"][key]["ce_attr"]
    return next((k for k, v in zip(KL, m) if v - base <= args.tol), C)


rng = np.random.default_rng(0)
order = sorted(present, key=lambda g: np.mean([thresh(g[0], d) for d in ds]))
for i, (key, col, _, _) in enumerate(order):
    v = np.array([thresh(key, d) for d in ds], dtype=float)
    ax.bar(i, v.mean(), 0.64, color=col, zorder=3)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.11, 0.11, len(v)),
               v, s=76 * S, color="white", edgecolors=INK,
               linewidths=1.35 * S, zorder=5)
    txt = f"{v.mean():.0f}" + ("*" if v.mean() >= C else "")
    ax.annotate(txt, (i, v.max()), (i, v.max() + C * 0.022), ha="center",
                fontsize=11 * S, fontweight="600", color=INK)
lab = {"eap_linmlp_gimattn": "EAP+linMLP\n+GIMattn", "gim": "GIM",
       "eapig_inputs5": "EAP-IG\ninputs", "eap_linmlp": "EAP\n+linMLP",
       "eap": "EAP", "ig5": "IG K=5", "gim_softmax_only": "GIM\nτ only",
       "random": "random\ndecomp"}
ax.set_xticks(range(len(order)))
ax.set_xticklabels([lab[k] for k, *_ in order], fontsize=8.6 * S, color=INK,
                   linespacing=1.4)
ax.set_xlim(-0.62, len(order) - 0.38)
ax.set_ylim(0, C * 1.42)
ax.axhline(C, color=INK, ls=(0, (6, 3)), lw=1.4 * S, zorder=4)
ax.annotate("* never gets there: needs all 256", xy=(0.5, 0.90),
            xycoords="axes fraction", ha="center", fontsize=9.4 * S, color=INK2)
ax.set_ylabel(f"components needed to get within {args.tol} nats  (of {C})",
              fontsize=10.5 * S, color=INK)
ax.set_title("Lower = fewer components reproduce the token",
             fontsize=12.5 * S, color=INK, pad=11)
ax.annotate(f"white dots = the {len(ds)} seeds", xy=(0.5, 0.965),
            xycoords="axes fraction", ha="center", fontsize=9.4 * S, color=INK2)

fig.suptitle(args.title, fontsize=15.5 * S, color=INK, x=0.05, ha="left",
             y=0.972)
fig.text(0.05, 0.905, args.note, fontsize=9.6 * S, color=MUTED, ha="left",
         va="top", linespacing=1.62)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(f"{args.out}{ext}", facecolor="white", **kw)
print(f"wrote {args.out}.png/.pdf/.svg  base={base:.4f} C={C} seeds={len(ds)}")
for key, *_ in present:
    v = np.array([thresh(key, d) for d in ds], dtype=float)
    rt = np.max([abs(d["results"][key]["roundtrip_err_at_C"]) for d in ds])
    print(f"  {key:<20} k needed: {v.mean():6.1f}   roundtrip |err| <= {rt:.1e}")
