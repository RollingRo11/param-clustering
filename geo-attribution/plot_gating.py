"""Attribution gating: does the token's own attribution pick better components?

Keep the k highest-scoring components at each position, ablate the rest, and
read the log-probability of the true next token. The comparison that decides
the question is `attr` against `shuffled` — the same attribution, taken from a
DIFFERENT position. If per-token attribution carried information about which
components this token needs, that gap would be large and consistent.
"""
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
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
STYLE = {
    "attr": ("#2a78d6", "o", "gated by THIS token's attribution"),
    "shuffled": ("#0b0b0b", "s", "gated by ANOTHER token's attribution"),
    "mass": ("#898781", "^", "gated by global weight mass"),
    "posterior": ("#eb6834", "D", "gated by fingerprint posterior"),
}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

d = json.loads((RUN / "attribution_gating.json").read_text())
R = d["results"]


def series(key):
    e = sorted(R[f"{key}_topk"], key=lambda r: r["k"])
    return (np.array([r["k"] for r in e], float),
            np.array([r["delta_logprob"] for r in e]))


def per(key, k):
    for e in R[f"{key}_topk"]:
        if e["k"] == k:
            return np.array([v for _, v in sorted(e["per_position"].items(),
                                                  key=lambda x: int(x[0]))])


def chrome(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9.5 * S, width=1.1, length=4)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.9 * S)


fig, axes = plt.subplots(1, 2, figsize=(17.4, 7.2), facecolor="white",
                         gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(top=0.63, bottom=0.125, left=0.058, right=0.985,
                    wspace=0.22)

ax = axes[0]
chrome(ax)
for key in ("posterior", "mass", "shuffled", "attr"):
    c, m, lab = STYLE[key]
    k, v = series(key)
    ax.plot(k, v, color=c, marker=m, linewidth=2.5 * S, markersize=6 * S,
            markeredgecolor="white", markeredgewidth=1.1 * S, label=lab,
            zorder=4 if key == "attr" else 3)
ax.axhline(0, color=INK, ls=(0, (4, 3)), lw=1.4 * S)
ax.annotate("unedited model", xy=(9, 1.8), fontsize=9.5 * S, color=INK)
ax.set_xscale("log")
ax.set_xlabel("components kept at each token  (of 4,096)", fontsize=10.5 * S,
              color=INK)
ax.set_ylabel("Δ log p(true next token)", fontsize=10.5 * S, color=INK)
ax.set_title("Keeping the top-k components at each position",
             fontsize=12.5 * S, color=INK, pad=11)
ax.legend(loc="lower right", frameon=False, fontsize=9.8 * S)

ax = axes[1]
chrome(ax)
ks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
gap = np.array([per("attr", k).mean() - per("shuffled", k).mean() for k in ks])
rng = np.random.default_rng(0)
ps = []
for k in ks:
    dif = per("attr", k) - per("shuffled", k)
    sg = rng.choice([-1, 1], size=(50000, len(dif)))
    ps.append((np.abs((sg * dif).mean(1)) >= abs(dif.mean())).mean())
cols = ["#2a78d6" if p < 0.05 else "#c3c2b7" for p in ps]
ax.bar(range(len(ks)), gap, 0.62, color=cols, zorder=3)
ax.axhline(0, color=INK, lw=1.5 * S)
for i, (g_, p) in enumerate(zip(gap, ps)):
    if p < 0.05:
        ax.annotate(f"p={p:.3f}", (i, g_), (i, g_ + 0.55), ha="center",
                    fontsize=9.2 * S, color="#2a78d6", fontweight="600")
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([str(k) for k in ks], fontsize=9.4 * S)
ax.set_xlabel("components kept", fontsize=10.5 * S, color=INK)
ax.set_ylabel("attr − shuffled  (nats, paired)", fontsize=10.5 * S, color=INK)
ax.set_title("Does the token's OWN attribution help?", fontsize=12.5 * S,
             color=INK, pad=11)
ax.annotate("above 0 = the token's own attribution\npicks better components\n\n"
            "grey = not significant (paired sign-flip,\n50k permutations, n=24 positions)",
            xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=9.3 * S, color=INK2, linespacing=1.5)

fig.suptitle("Attribution gating — scaling components by their attribution to the token",
             fontsize=15 * S, color=INK, x=0.058, ha="left", y=0.972)
fig.text(0.058, 0.907,
         "Attribution is the GIM sensor the decomposition was built from: a single modified backward "
         "(τ=2 softmax Jacobian, scaling/4 on Q and K, halved\n"
         "gate/up gradients, frozen RMSNorm statistics) against a PRE-softmax objective that puts one "
         "unit of reward on each ground-truth next-token logit —\n"
         "not plain gradients and not integrated gradients. A component's score at token t is "
         "Σ|∂Reward/∂W · W| over the entries it owns. Weights are shared\n"
         "across positions, so this is one weight rebuild and one forward per position: 24 positions × "
         "46 configurations in 168 s.",
         fontsize=9.6 * S, color=MUTED, ha="left", va="top", linespacing=1.6)

for ext, kw in ((".png", {"dpi": 400}), (".pdf", {}), (".svg", {})):
    fig.savefig(GEO / f"out/attribution_gating{ext}", facecolor="white", **kw)
print("wrote out/attribution_gating.png/.pdf/.svg")
for k, g_, p in zip(ks, gap, ps):
    print(f"  k={k:<5} attr-shuffled {g_:+7.3f}  p={p:.4f}")
