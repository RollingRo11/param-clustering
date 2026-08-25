import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("out")
canon = json.loads((D / "klkeep.json").read_text())
oracle = json.loads((D / "klkeep_oracle.json").read_text())
vpd = json.loads((D / "klkeep_vpd.json").read_text())

C_OURS = 1024
xs_o = [int(k) / C_OURS * 100 for k in sorted(canon, key=int)]
ko = sorted(canon, key=int)
xs_v = sorted(float(p) for p in vpd)

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(xs_o, [canon[k]["kl_mean"] for k in ko], lw=3.5, marker="o", ms=6,
        color="#c9b6f7", label="co-fac canonical (C=1024)")
ax.plot(xs_o, [oracle[k]["kl_mean"] for k in ko], lw=1.6, marker="s", ms=3,
        color="#5b21b6", ls=(0, (3, 2)), label="co-fac oracle — coincides")
ax.plot(xs_o, [canon[k]["kl_rand_mean"] for k in ko], lw=1.8, marker="o",
        ms=3, color="#b3b1a8", label="co-fac random null")
ax.plot(xs_v, [vpd[str(p) if str(p) in vpd else str(int(p))]["kl_mean"]
               for p in xs_v], lw=2.2, marker="o", ms=5, color="#1baf7a",
        label="VPD canonical (CI order, 38,912 subcomps)")
ax.plot(xs_v, [vpd[str(p) if str(p) in vpd else str(int(p))]["kl_rand_mean"]
               for p in xs_v], lw=1.8, marker="o", ms=3, color="#6fcfae",
        ls=(0, (2, 2)), label="VPD random null")
ax.set_xscale("log", base=2)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_xlabel("percent of components kept per token")
ax.set_ylabel("KL to full model (nats)")
ax.set_xticks(xs_o)
ax.set_xticklabels([f"{x:.3g}%" for x in xs_o], fontsize=8)
ax.set_title("KLKeep on the 67M 4L-Pile model: co-fac vs VPD")
ax.grid(axis="y", color="#e8e8e8", lw=0.75)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8)
out = Path("figures/klkeep67_vs_vpd.png")
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
