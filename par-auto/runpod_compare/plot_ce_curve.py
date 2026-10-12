"""CE degradation from the intact baseline (and KL) versus pieces kept per token."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RUNS = [('parauto_1b', 'par-auto 1B', 'tab:blue', '-'),
        ('parauto_gim_5k', 'par-auto GIM 5k', 'tab:cyan', '-'),
        ('parauto_gim_13b_36k', 'par-auto GIM 13B run, step 36k (4.7B tokens)', 'tab:green', '-'),
        ('vpd_5k', 'VPD 5k', 'tab:orange', '--'),
        ('vpd_0.6b', 'VPD 0.6B (stopped early)', 'tab:red', '--'),
        ('vpd_13b', 'VPD 13B (published)', 'tab:purple', '--')]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
for key, label, colour, style in RUNS:
    if not (SRC/key/'klkeep.json').exists(): continue
    d = json.loads((SRC/key/'klkeep.json').read_text())
    rows = [r for r in d['curve'] if r['kept_components'] >= 1]
    kept = [r['kept_components'] for r in rows]
    axes[0].plot(kept, [r['ce']-d['baseline_ce'] for r in rows], color=colour, ls=style, marker='o', ms=3, label=label)
    axes[1].plot(kept, [r['kl'] for r in rows], color=colour, ls=style, marker='o', ms=3, label=label)
for ax, title, ylabel in ((axes[0], 'CE degradation', 'CE increase over the intact model (nats/token)'),
                          (axes[1], 'KL, same runs', 'KL(intact || masked) (nats/token)')):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('pieces kept per token')
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.3, which='both')
axes[0].legend(fontsize=8, loc='lower left')
fig.suptitle('Same 128 held-out sequences; each method ranked by its own selector '
             '(ours Fisher K=8, VPD causal importance)', fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f'wrote {OUT}')
