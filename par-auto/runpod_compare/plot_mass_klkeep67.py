"""Two panels: per-piece weight mass (sorted) and KLKeep, for the matryoshka67 comparison checkpoints.
Usage: plot_mass_klkeep67.py MASS_JSON OUTPUT_PNG"""
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
EXP = ROOT/'output/matryoshka67_20260923'
# Categorical slots 1-5 of the validated reference palette, fixed order; new methods solid, baselines dashed.
SERIES = [
    ('matryoshka_hard_5000', 'new, hard forward (5k)', '#2a78d6', '-', EXP/'main_hard/eval_fisher_k8/klkeep.json'),
    ('matryoshka_soft_5000', 'new, soft forward (5k)', '#eb6834', '-', EXP/'main/eval_fisher_k8/klkeep.json'),
    ('fisher_k4_5000', 'Fisher baseline (5k)', '#1baf7a', '--', ROOT/'output/fisher_k4_5000/eval_fisher_k8/klkeep.json'),
    ('vpd_5000', 'VPD (5k)', '#eda100', '--', ROOT/'output/evaluations/gim_normal_vpd_5000/vpd/klkeep.json'),
    ('vpd_published_400000', 'VPD published (400k)', '#e87ba4', '--', ROOT/'output/published_vpd_vs_finished_parauto/vpd/klkeep.json'),
]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    mass = json.loads(Path(sys.argv[1]).read_text())['methods']
    output = Path(sys.argv[2])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=170, facecolor=SURFACE)
    for label, name, color, style, klkeep in SERIES:
        m = mass[label]
        shares = 100*np.clip(np.array(m['sorted_shares']), 1e-12, None)
        left.plot(np.arange(1, len(shares)+1), shares, color=color, ls=style, lw=2,
                  label=f"{name}: pieces total {100*m['total_share']:.0f}%")
        curve = [r for r in json.loads(klkeep.read_text())['curve'] if r['kept_components'] > 0]
        right.plot([r['kept_components'] for r in curve], [r['kl'] for r in curve], color=color, ls=style, lw=2,
                   marker='o', ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1, label=name)
    left.set_yscale('log'); left.set_xscale('log')
    left.set_ylim(bottom=1e-6)      # below this are the Fisher baseline's empty pieces (the 1e-12 clip, not data)
    left.set_xlabel('pieces, sorted by weight mass', color=INK)
    left.set_ylabel('% of model weight mass (per piece)', color=INK)
    left.set_title('Weight mass per piece', color=INK, loc='left', fontsize=11)
    left.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='lower left')
    right.set_xscale('log'); right.set_yscale('log')
    right.set_xlabel('pieces kept per token', color=INK)
    right.set_ylabel('KL to the intact model', color=INK)
    right.set_title('KLKeep (65,536 held-out positions)', color=INK, loc='left', fontsize=11)
    right.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='lower left')
    for ax in (left, right):
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
