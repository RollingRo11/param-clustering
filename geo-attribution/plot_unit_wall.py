"""How much of the 67M model every geo run must keep to reach KL < 1, vs VPD.

One dot per geo run (unitabl_eval_*.json, 48 held-out tokens, real masked
forwards), under three removal orders: random, the run's own attribution
ranking, and the oracle (true one-at-a-time removal effect). Duplicate runs
(identical curves) are dropped. VPD: vpd_match67.json, CI order, on the
same 48 tokens (stream67 phase_bank selection).

    uv run --with matplotlib --with numpy python plot_unit_wall.py OUT.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "out"
GEO, VPD = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"
ARMS = [("all/random", "random\norder"), ("all/attr", "geo's own\nranking"),
        ("all/oracle", "best possible\norder (oracle)")]


def first_below(kl, frac, thr=1.0):
    return next((f for k, f in zip(kl, frac) if k < thr), None)


runs, seen = {}, set()
for f in sorted(OUT.glob("unitabl_eval_*.json")):
    res = json.loads(f.read_text()).get("results", {})
    vals = {arm: first_below(res[arm]["kl"], res[arm]["units_kept_frac"])
            for arm, _ in ARMS if arm in res}
    key = tuple(round(v, 4) for v in vals.values() if v is not None)
    if "all/attr" in vals and key not in seen:
        seen.add(key)
        runs[f.stem] = vals
print(f"{len(runs)} distinct geo runs")

vpd = json.loads((OUT / "vpd_match67.json").read_text())
vpd_pct = first_below(vpd["results"]["keep"], vpd["pct"])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
rng = np.random.default_rng(0)
for i, (arm, _) in enumerate(ARMS):
    ys = np.array([100 * r[arm] for r in runs.values() if r.get(arm) is not None])
    ax.scatter(i + rng.uniform(-.16, .16, len(ys)), ys, s=34, color=GEO,
               edgecolor="white", linewidth=1, zorder=3,
               label=f"geo runs (n={len(runs)})" if i == 0 else None)
    lo = ys.min()
    ax.text(i + .24, np.median(ys), f"median {np.median(ys):.0f}%\nmin {lo:.0f}%",
            va="center", fontsize=8.5, color=MUTED)
ax.scatter([3], [vpd_pct], s=60, color=VPD, edgecolor="white", linewidth=1,
           zorder=3, label="VPD")
ax.text(3.2, vpd_pct, f"{vpd_pct:.1f}%", va="center", fontsize=8.5, color=MUTED)

ax.set_yscale("log")
ax.set_ylim(0.3, 130)
ax.set_yticks([0.5, 1, 5, 10, 50, 100])
ax.set_yticklabels(["0.5%", "1%", "5%", "10%", "50%", "100%"])
ax.set_xticks(range(4))
ax.set_xticklabels([lab for _, lab in ARMS] + ["VPD\n(its own ranking)"])
ax.set_xlim(-.5, 3.7)
ax.set_ylabel("% of the model kept to reach KL < 1", color=INK)
ax.tick_params(colors=MUTED)
ax.grid(axis="y", alpha=.25, linewidth=.6)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig(sys.argv[1], bbox_inches="tight")
print("wrote", sys.argv[1])
