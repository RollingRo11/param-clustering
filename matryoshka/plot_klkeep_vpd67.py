"""KLKeep of the Matryoshka lambda = 1 run against VPD at three training lengths (5k steps, the paired run's final
checkpoint, the published run).  Same 65,536 held-out positions, each method ranked by its own score.
Usage: plot_klkeep_vpd67.py OUTPUT_PNG"""
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
EXP = ROOT/'output/matryoshka67_20260923'
# Entities keep the colors of the 2026-09-26/27 figures; the paired VPD run takes green (validated all-pairs),
# and every VPD curve is dashed so it cannot be confused with the green lambda = 0.1 line of 2026-09-27.
SERIES = [
    ('new, hard, λ = 1', '#4a3aa7', '-', 2.6, EXP/'main_hard_lp1/eval_fisher_k8/klkeep.json'),
    ('VPD 5k', '#eda100', '--', 2, ROOT/'output/evaluations/gim_normal_vpd_5000/vpd/klkeep.json'),
    ('VPD paired', '#008300', '--', 2, ROOT/'output/ce_curve_20260916/vpd_0.6b/klkeep.json'),
    ('VPD published', '#e87ba4', '--', 2, ROOT/'output/published_vpd_vs_finished_parauto/vpd/klkeep.json'),
]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    output = Path(sys.argv[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=170, facecolor=SURFACE)
    data_sha = set()
    for name, color, style, width, path in SERIES:
        result = json.loads(path.read_text())
        assert result['positions'] == 65536, (path, result['positions'])
        data_sha.add(result['data_sha256'])
        curve = [r for r in result['curve'] if r['kept_components'] > 0]
        label = f"{name} ({result['training_steps']:,} steps, {result['training_positions'] / 1e9:.2f}B tokens)"
        ax.plot([r['kept_components'] for r in curve], [r['kl'] for r in curve], color=color, ls=style, lw=width,
                marker='o', ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    assert len(data_sha) == 1, data_sha
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('pieces kept per token', color=INK)
    ax.set_ylabel('KL to the intact model', color=INK)
    ax.set_title('KLKeep: new method (λ = 1) vs VPD at three training lengths (65,536 held-out positions)',
                 color=INK, loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower left')
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)


if __name__ == '__main__': main()
