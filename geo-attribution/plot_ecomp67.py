"""Keep and drop curves for the E-clustered unit components (ecomp67), with
VPD on the same axes.

x axis is the number of components in the top-k set: our components, or
VPD's rank-1 subcomponents, shown up to its ALIVE count (mean causal
importance > 1e-6 on the 1M corpus, vpd_alive67).  VPD keep/drop come from
vpd_match67 (same 48 tokens).

Writes two PNGs (keep, drop) into the obsidian Attachments folder, named by
date so the daily note can embed them.

    uv run --no-project --with matplotlib python plot_ecomp67.py \
        out/unitabl_eval_1m_fine_e1024.json 2026-09-05
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ATTACH = Path("/Users/rohan/Developer/obsidian/Attachments")
OUT = Path(__file__).parent / "out"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})
OURS = {"oracle": ("ours, measured effect ranking", "#1b7f79", "-"),
        "attr": ("ours, first-order attribution ranking", "#b4531f", "-")}
VPD_COL = "#6a3d9a"


def main():
    path = Path(sys.argv[1])
    date = sys.argv[2]
    d = json.load(open(path))
    C = int(d["C"])
    res = {k[4:]: v for k, v in d["results"].items() if k.startswith("all/")}
    tag = path.stem.replace("unitabl_eval_", "")

    vm = json.load(open(OUT / "vpd_match67.json"))
    va = json.load(open(OUT / "vpd_alive67.json"))
    n_alive = va["corpora"]["lab_1m"]["alive_mean_ci_clamp"]["1e-06"]
    vk = vm["vpd_k"]
    keep_v = [i for i, k in enumerate(vk) if k <= n_alive]      # up to 100% of alive
    vx = [vk[i] for i in keep_v]

    for kind, key, vkey, title in [
            ("keep", "kl", "keep", f"Keep the top-k per token (ours: {C} components)"),
            ("drop", "kl_drop", "drop", f"Drop the top-k per token (ours: {C} components)")]:
        fig, ax = plt.subplots(figsize=(6.6, 4.1))
        for name in ("oracle", "attr"):
            r = res[name]
            # drop: stop at 90% of units removed (beyond that the model is simply off)
            idx = [i for i, k in enumerate(r["keep_grid"])
                   if k > 0 and (kind == "keep" or (k < C and r["units_kept_frac"][i] <= 0.9))]
            xs = [r["keep_grid"][i] for i in idx]
            ys = [r[key][i] for i in idx]
            lab, col, ls = OURS[name]
            ax.plot(xs, ys, ls, color=col, lw=1.8, label=lab)
        ax.plot(vx, [vm["results"][vkey][i] for i in keep_v], "-", color=VPD_COL,
                lw=1.8, marker="o", ms=3, label="VPD, causal-importance ranking")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("k components in the top-k set  (ours: components;  "
                      f"VPD: rank-1 subcomponents, up to its {n_alive:,} alive)", fontsize=8.5)
        ax.set_ylabel("KL to the full model (nats)")
        ax.set_title(title, fontsize=10.5)
        if kind == "keep":
            ax.legend(loc="upper right", fontsize=8.5)
        else:
            ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        out = ATTACH / f"{date} ecomp {tag} {kind}.png"
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
