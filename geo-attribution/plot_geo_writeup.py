"""Figures for the CBAI geo-attribution writeup (67M pile-4L target).

Inputs are result JSONs already on disk in geo-attribution/out; outputs land
next to the writeup in the obsidian vault (override with GEO_WRITEUP_FIGS).

    uv run --with matplotlib --with numpy python plot_geo_writeup.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/Users/rohan/Developer/param-clustering/geo-attribution/out")
FIG = Path(os.environ.get(
    "GEO_WRITEUP_FIGS",
    "/Users/rohan/Developer/obsidian/Research/CBAI/figures"))
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
})

GEO, GEO2, VPD, GREY = "#1b7f79", "#7fbfbb", "#b4531f", "#8a8a8a"

# ------------------------------------------------------------------ data ----
s4k = json.loads((OUT / "bank_C4096_ig5_t200m.json").read_text())
C4 = s4k["C"]
grid = np.array(s4k["keep_grid"], dtype=float)
arms = s4k["results"]
base_ce = s4k["base_token_ce"]

# C=8192 arm (same pilot/seed/spec/eval tokens as C=4096; present once the
# cluster job lands, absent before then)
S8K = OUT / "bank_C8192_ig5_t200m.json"
s8k = json.loads(S8K.read_text()) if S8K.exists() else None

vpd_c = json.loads((OUT / "klkeep_vpd_big.json").read_text())
vpd_f = json.loads((OUT / "klkeep_vpd_big_fine.json").read_text())
vpd = {**{float(k): v for k, v in vpd_f.items()},
       **{float(k): v for k, v in vpd_c.items()}}
vp = np.array(sorted(p for p in vpd if 0 < p < 100))
vkl = np.array([vpd[p]["kl_mean"] for p in vp])
vkl_r = np.array([vpd[p]["kl_rand_mean"] for p in vp])
N_VPD = 38912

# ------------------------------------- fig 1: keep curve, percent-of-units ---
fig, ax = plt.subplots(figsize=(7.2, 4.4))
pct = grid[1:-1] / C4 * 100
for name, style, col, lw, lab in [
        ("refined", "-", GEO, 2.0, "geo, refined shares (C=4096)"),
        ("original", "--", GEO2, 1.6, "geo, proportional shares (C=4096)"),
        ("random", ":", GREY, 1.4, "geo, random-k null")]:
    kl = np.array(arms[name]["kl"])[1:-1]
    ax.plot(pct, kl, style, color=col, lw=lw, label=lab,
            marker="o" if name == "refined" else None, ms=3)
if s8k is not None:
    g8 = np.array(s8k["keep_grid"], dtype=float)
    k8 = np.array(s8k["results"]["refined"]["kl"])
    m8 = (g8 > 0) & (g8 < s8k["C"])
    ax.plot(g8[m8] / s8k["C"] * 100, k8[m8], "-", color="#3f5fa8", lw=2.0,
            marker="^", ms=3, label=f"geo, refined shares (C={s8k['C']})")
ax.plot(vp, vkl, "-", color=VPD, lw=2.0, marker="s", ms=3,
        label="VPD, causal-importance rank (38,912 subcomponents)")
ax.plot(vp, vkl_r, ":", color="#d9a68a", lw=1.4, label="VPD, random-k null")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("units kept, % of the method's own component count (log)")
ax.set_ylabel("KL to the full model at the held-out token, nats (log)")
ax.set_title("Keep-top-k ablation: attribution geometry vs VPD, 67M pile-4L")
ax.set_ylim(0.35, 90)
ax.annotate(f"geo floor {arms['refined']['kl'][7]:.1f} nats\nat k=192 (4.7% of C)",
            xy=(192 / C4 * 100, arms["refined"]["kl"][7]),
            xytext=(0.42, 0.22), textcoords="axes fraction",
            fontsize=8.5, color=GEO, ha="center",
            arrowprops=dict(arrowstyle="-", color=GEO, lw=0.8))
ax.annotate("VPD floor 0.93 nats\nat 0.78% (304 units)",
            xy=(0.78125, 0.9296), xytext=(0.26, 0.03),
            textcoords="axes fraction", fontsize=8.5, color=VPD, ha="center",
            arrowprops=dict(arrowstyle="-", color=VPD, lw=0.8))
ax.legend(loc="upper right", fontsize=8.2)
fig.tight_layout(); fig.savefig(FIG / "geo_keep_kl_pct.png"); plt.close(fig)

# ------------------------------------------- fig 2: CE sufficiency curve -----
fig, ax = plt.subplots(figsize=(7.2, 4.2))
k = grid[1:]
for name, style, col, lw, lab in [
        ("refined", "-", GEO, 2.0, "refined shares"),
        ("original", "--", GEO2, 1.6, "proportional shares"),
        ("random", ":", GREY, 1.4, "random-k null")]:
    ax.plot(k, np.array(arms[name]["ce"])[1:], style, color=col, lw=lw,
            marker="o" if name == "refined" else None, ms=3, label=lab)
ax.axhline(base_ce, color="#333", lw=1.0, ls="-.")
ax.text(4.4, base_ce * 1.12, f"full model CE = {base_ce:.2f}", fontsize=8.5)
ax.axvline(192, color=GEO, lw=0.8, alpha=0.5)
ax.text(205, 12, "k = 192\n(CE within 0.25 nats)", fontsize=8.5, color=GEO)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("components kept, out of C = 4096 (log)")
ax.set_ylabel("cross-entropy at the held-out token, nats (log)")
ax.set_title("Same run, CE view: 192 of 4096 components beat the full model's CE")
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "geo_keep_ce.png"); plt.close(fig)

# --------------------------------------------- fig 3: token-budget sweep -----
SW = OUT / "sweep67"
runs = [("bank_t100k.json", "0.1M tokens"), ("bank_t10m.json", "10M"),
        ("bank_t100m.json", "100M"), ("bank_t500m.json", "500M"),
        ("bank_t1000m.json", "1B")]
cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0))
toks, k025, filled = [], [], []
for (name, lab), c in zip(runs, cols):
    d = json.loads((SW / name).read_text())
    g = np.array(d["keep_grid"], dtype=float)
    ce = np.array(d["results"]["refined"]["ce"])
    axs[0].plot(g[1:], ce[1:], "-", color=c, lw=1.7, label=lab)
    toks.append(d["tokens_streamed"])
    k025.append(d["results"]["refined"]["k_within_0.25"])
    filled.append(100 * d["slots_filled"] / d["slots_total"])
    Cw = d["C"]
axs[0].axhline(d["base_token_ce"], color="#333", lw=1.0, ls="-.")
axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel(f"components kept, out of C = {Cw} (log)")
axs[0].set_ylabel("CE at the held-out token, nats (log)")
axs[0].set_title("Streaming budget sweep (C = 8192)")
axs[0].legend(fontsize=8.5, title="streamed", title_fontsize=8.5)
ax2 = axs[1]
ax2.plot(toks, filled, "-o", color=GEO, lw=1.8, ms=4,
         label="reservoir slots filled (%)")
ax2.set_xscale("log"); ax2.set_xlabel("attributed tokens streamed (log)")
ax2.set_ylabel("reservoir slots filled, %", color=GEO)
ax2.tick_params(axis="y", labelcolor=GEO); ax2.set_ylim(0, 105)
ax3 = ax2.twinx()
ax3.plot(toks, k025, "-s", color=VPD, lw=1.8, ms=4)
ax3.set_ylabel("components needed for CE within 0.25 nats", color=VPD)
ax3.tick_params(axis="y", labelcolor=VPD); ax3.grid(False)
ax3.set_ylim(0, 420)
ax2.annotate("92% of clusters\nstill empty here", xy=(toks[0], filled[0]),
             xytext=(0.30, 0.05), textcoords="axes fraction",
             fontsize=8.2, color=GREY,
             arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
ax2.set_title("More tokens fill the reservoir; they do not shrink k")
fig.tight_layout(); fig.savefig(FIG / "geo_token_sweep.png"); plt.close(fig)

# ------------------------------------------------- fig 4: mass Lorenz -------
mg = json.loads((OUT / "mass_stream4k.json").read_text())
mv = json.loads((OUT / "vpd_mass_dist.json").read_text())["fro_share"]
mr8 = json.loads((OUT / "mass_r8.json").read_text())


def lorenz(shares):
    s = np.sort(np.asarray(shares, dtype=float))[::-1]
    s = s / s.sum()
    return np.arange(1, len(s) + 1) / len(s), np.cumsum(s)


fig, ax = plt.subplots(figsize=(6.6, 4.2))
x, y = lorenz(mg["shares_sorted"])
ax.plot(100 * x, 100 * y, "-", color=GEO, lw=2.0,
        label=f"geo components (C={mg['C']}, zero base)")
M8K = OUT / "mass_stream8k.json"
if M8K.exists():
    m8 = json.loads(M8K.read_text())
    x, y = lorenz(m8["shares_sorted"])
    ax.plot(100 * x, 100 * y, "-", color="#3f5fa8", lw=1.8,
            label=f"geo components (C={m8['C']}, zero base)")
x, y = lorenz(mv)
ax.plot(100 * x, 100 * y, "-", color=VPD, lw=2.0,
        label=f"VPD subcomponents (n={len(mv):,})")
x, y = lorenz(mr8["comp_shares_sorted"])
ax.plot(100 * x, 100 * y, "--", color=GEO2, lw=1.6,
        label=f"geo C=512 partition, of the {100*(1-mr8['base_share']):.0f}% "
              "outside its base")
ax.plot([0, 100], [0, 100], color=GREY, lw=0.9, ls=":", label="equal shares")
ax.set_xlabel("units, ranked by owned weight mass (%)")
ax.set_ylabel("cumulative share of total $\\|W\\|_F^2$ (%)")
ax.set_title("Weight-mass concentration: no mega-component on either side")
ax.legend(loc="lower right", fontsize=8.2)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
fig.tight_layout(); fig.savefig(FIG / "geo_mass_lorenz.png"); plt.close(fig)

# ------------------------------------------- fig 5: base-mass frontier ------
# N=65k, C=512 hard-partition arms (intermediate.md section 12 table)
r = [2, 3, 4, 5, 8, 12, 16]
base = [0.000, 0.163, 0.459, 0.669, 0.932, 0.984, 0.990]
top256 = [3.67, 3.01, 1.72, 0.65, 0.067, 0.008, 0.002]
fresh = json.loads((OUT / "eval_r8.json").read_text())
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.plot(100 * np.array(base), top256, "-o", color=GEO, lw=1.8, ms=5)
for ri, b, t in zip(r, base, top256):
    ax.annotate(f"r={ri}", (100 * b, t), textcoords="offset points",
                xytext=(6, 5), fontsize=8.5, color=GEO)
ax.plot([100 * fresh["resid_frac_mean"]], [fresh["keep_top_j"]["256"]], "x",
        color=VPD, ms=9, mew=2,
        label=f"r=8 re-run 2026-08-27 (N=32k): base "
              f"{100*fresh['resid_frac_mean']:.0f}%, "
              f"KL {fresh['keep_top_j']['256']:.2f}")
ax.axhline(2.55, color=GEO2, lw=1.2, ls="--")
ax.text(50, 2.9, "zero-base proportional shares (C=512): 2.55",
        fontsize=8.3, color=GEO2, ha="center")
ax.set_yscale("log")
ax.set_xlabel("weight mass parked in the always-on base, %")
ax.set_ylabel("keep-top-256 KL, nats (log)")
ax.set_title("Why the base is a confound: fidelity is bought with owned mass")
ax.legend(loc="lower left", fontsize=8.2)
fig.tight_layout(); fig.savefig(FIG / "geo_base_frontier.png"); plt.close(fig)

print("wrote:", *[p.name for p in sorted(FIG.glob("*.png"))])
