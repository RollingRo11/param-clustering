import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("out")
canon = json.loads((D / "klkeep.json").read_text())
oracle = json.loads((D / "klkeep_oracle.json").read_text())
ks = sorted(int(k) for k in canon)
fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.plot(ks, [canon[str(k)]["kl_mean"] for k in ks], lw=3.5, ms=6,
        marker="o", color="#c9b6f7", solid_capstyle="round",
        label="canonical (top-|z| per token)")
ax.plot(ks, [oracle[str(k)]["kl_mean"] for k in ks], lw=1.6, ms=3,
        marker="s", color="#5b21b6", ls=(0, (3, 2)),
        label="oracle (true ablation order) — coincides")
ax.plot(ks, [canon[str(k)]["kl_rand_mean"] for k in ks], "-o", lw=2, ms=4,
        color="#9a9890", label="random-k null")
ax.set_xscale("log", base=2)
ax.set_yscale("symlog", linthresh=1e-3)
ax.set_xlabel("components kept per token (of 1024)")
ax.set_ylabel("KL to full model (nats)")
ax.set_title("KLKeep on the 67M 4L-Pile model (co-fac, C=1024)")
ax.grid(axis="y", color="#e8e8e8", lw=0.75)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8.5)
out = Path("figures/klkeep67_oracle_canonical.png")
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
