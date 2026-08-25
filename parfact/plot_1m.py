import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).parent / "out/curves_v2"
lo = json.loads((D / "curves_logodds.json").read_text())
us = json.loads((D / "curves_v2_usimplex.json").read_text())
base = {t: json.loads((D / f"curves_{t}.json").read_text())
        for t in ("v1", "v2")}
sp = json.loads((D / "curves_spd1m.json").read_text())
lv = json.loads((D / "curves_lv_sweep.json").read_text())["curves"]["0.0001"]

series = [
    ("co-fac v1", "#2a78d6", lo["v1"]["curves"]["canonical"],
     base["v1"]["curves"]["oracle"]),
    ("co-fac v2", "#8a3ffc", lo["v2"]["curves"]["canonical"],
     base["v2"]["curves"]["oracle"]),
    ("co-fac v2 + U-simplex", "#d4437c", us["curves"]["canonical"],
     us["curves"]["oracle"]),
    ("co-fac v2 + U-simplex + λ_V", "#e0862c",
     lv["curves"]["canonical"], lv["curves"]["oracle"]),
    ("SPD (their code, 1M steps)", "#1baf7a", sp["curves"]["canonical"],
     sp["curves"]["oracle"]),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for name, color, canon, oracle in series:
    ax.plot([r["k"] for r in canon], [r["acc"] for r in canon],
            lw=2, color=color, label=f"{name} — canonical")
    ax.plot([r["k"] for r in oracle], [r["acc"] for r in oracle],
            lw=2, color=color, ls=(0, (2, 2)), label=f"{name} — oracle")
ax.axhline(1 / 128, color="#999999", lw=1, ls=(0, (4, 3)))
ax.set_xlabel("components ablated per token (of 600, least important first)")
ax.set_ylabel("induction accuracy")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Keep-top-k minimality at C=600")
ax.grid(axis="y", color="#e8e8e8", lw=0.75)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
out = Path(__file__).parent / "figures/keep_topk_with_spd1m.png"
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
