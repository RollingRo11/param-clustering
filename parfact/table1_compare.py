"""Remake Christensen & Riggs Table 1 with OUR replication numbers (black)
against their reported values (gray), for the C=600 SPD run spd_C600."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ours: spd_C600_concat -- Table-4 losses AND the paper's gate (attn (+) x_n),
# g>0.5 census over 2048 seqs; theirs: their Table 1
ROWS = ["Q0", "K0", "V0", "Q1", "K1", "V1"]
OURS_A = {"Q0": (0.02, 1.67, 0.06, 0.00), "K0": (2.13, 0.11, 0.00, 0.00),
          "V0": (0.96, 0.14, 0.00, 0.00), "Q1": (0.00, 0.00, 1.00, 0.00),
          "K1": (0.02, 0.99, 0.02, 0.00), "V1": (0.00, 4.04, 0.22, 0.00)}
THEIRS_A = {"Q0": (0.000, 1.000, 0.000, 0.001),
            "K0": (1.000, 0.050, 0.000, 0.183),
            "V0": (1.000, 0.000, 0.000, 0.000),
            "Q1": (0.000, 0.000, 1.000, 0.000),
            "K1": (0.000, 1.000, 0.000, 0.000),
            "V1": (0.000, 5.053, 0.000, 0.000)}
OURS_B = {"Q0": 15, "K0": 21, "V0": 37, "Q1": 1, "K1": 6, "V1": 81}
THEIRS_B = {"Q0": 1, "K0": 1, "V0": 1, "Q1": 1, "K1": 1, "V1": 11}
METRICS = [("L_faithful", "1.6e-10", "3e-9"),
           ("L_recon (plain)", "~0", "1e-4"),
           ("L_stoch_recon (avg32)", "2.9e-4", "-"),
           ("L_stoch_recon_lw (avg32)", "1.1e-4", "1e-4"),
           ("total unique", "161", "16")]

BLACK, GRAY, RULE = "#111111", "#9a9890", "#333333"
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), dpi=170,
                         gridspec_kw={"width_ratios": [1.5, 0.85, 1.15]})
fig.patch.set_facecolor("white")

def rule(ax, y, x0=0.02, x1=0.98, lw=1.1):
    ax.plot([x0, x1], [y, y], color=RULE, lw=lw, clip_on=False)

def panel_a(ax):
    cols = ["s1", "m", "s2", "random"]
    xs = [0.28, 0.46, 0.64, 0.85]
    ax.text(0.12, 0.93, "", fontsize=9)
    for x, c in zip(xs, cols):
        ax.text(x, 0.90, c, ha="center", fontsize=10.5, style="italic",
                color=BLACK)
    rule(ax, 0.965); rule(ax, 0.955, lw=0.6); rule(ax, 0.865)
    y = 0.795
    for r in ROWS:
        ax.text(0.08, y, r, fontsize=10.5, color=BLACK, style="italic")
        for x, ov, tv in zip(xs, OURS_A[r], THEIRS_A[r]):
            ax.text(x, y, f"{ov:.2f}", ha="center", fontsize=10.5,
                    color=BLACK)
            ax.text(x, y - 0.052, f"{tv:.3f}", ha="center", fontsize=7,
                    color=GRAY)
        if r == "V0":
            rule(ax, y - 0.085, lw=0.7)
        y -= 0.125
    rule(ax, y + 0.055); rule(ax, y + 0.045, lw=0.6)
    ax.text(0.5, y - 0.01, "(a)  Average active subcomponents\n"
            "(ours, g>0.5; gray = Christensen & Riggs)",
            ha="center", va="top", fontsize=9.5, color=BLACK)

def panel_b(ax):
    ax.text(0.68, 0.90, "Total unique", ha="center", fontsize=10.5,
            color=BLACK)
    rule(ax, 0.965); rule(ax, 0.955, lw=0.6); rule(ax, 0.865)
    y = 0.795
    for r in ROWS:
        ax.text(0.16, y, r, fontsize=10.5, color=BLACK, style="italic")
        ax.text(0.62, y, str(OURS_B[r]), ha="center", fontsize=10.5,
                color=BLACK, fontweight="bold" if r == "V1" else "normal")
        ax.text(0.85, y, f"({THEIRS_B[r]})", ha="center", fontsize=8,
                color=GRAY)
        if r == "V0":
            rule(ax, y - 0.085, lw=0.7)
        y -= 0.125
    rule(ax, y + 0.055); rule(ax, y + 0.045, lw=0.6)
    ax.text(0.5, y - 0.01, "(b)  Total unique subcomponents\n"
            "(ours; gray = theirs). Total: 161 vs 16",
            ha="center", va="top", fontsize=9.5, color=BLACK)

def panel_c(ax):
    ax.text(0.08, 0.90, "Metric", fontsize=10.5, color=BLACK)
    ax.text(0.60, 0.90, "Ours", ha="center", fontsize=10.5, color=BLACK)
    ax.text(0.86, 0.90, "Theirs", ha="center", fontsize=10.5, color=GRAY)
    rule(ax, 0.965); rule(ax, 0.955, lw=0.6); rule(ax, 0.865)
    y = 0.795
    for name, ov, tv in METRICS:
        ax.text(0.08, y, name, fontsize=10, color=BLACK)
        ax.text(0.60, y, ov, ha="center", fontsize=10, color=BLACK)
        ax.text(0.86, y, tv, ha="center", fontsize=8.5, color=GRAY)
        y -= 0.125
    rule(ax, y + 0.055); rule(ax, y + 0.045, lw=0.6)
    ax.text(0.5, y - 0.01,
            "(c)  Exact losses on 2048-seq eval batch (stochastic\n"
            "terms averaged over 32 mask draws). D_KL-Attn rows\n"
            "omitted: undefined in the paper. Position selectivity\n"
            "REPLICATES; concentration does NOT (V1: 81 vs 11,\n"
            "robust over 4 independent runs).",
            ha="center", va="top", fontsize=8.8, color=BLACK)

for ax, fn in zip(axes, (panel_a, panel_b, panel_c)):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fn(ax)

fig.suptitle("Table 1, replicated (run spd_C600_concat: Table-4 losses + "
             "paper CI gate) — ours (black) vs Christensen & Riggs (gray)",
             fontsize=12, y=0.99)
out = Path("figures/table1_ours_vs_theirs.png")
out.parent.mkdir(exist_ok=True)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
