"""KLKeep of the Matryoshka runs with the Lp term (lambda 0, 0.1, 1) against the Fisher baseline and VPD.
Same 65,536 held-out positions, each method ranked by its own score.  Usage: plot_klkeep_lp67.py OUTPUT_PNG"""
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
EXP = ROOT/'output/matryoshka67_20260923'
# Validated reference palette; entities keep the colors of the 2026-09-26 figure, new runs take new slots.
SERIES = [
    ('new, hard, λ = 1', '#4a3aa7', '-', 2.6, EXP/'main_hard_lp1/eval_fisher_k8/klkeep.json'),
    ('new, hard, λ = 0.1', '#008300', '-', 2, EXP/'main_hard_lp0.1/eval_fisher_k8/klkeep.json'),
    ('new, hard, λ = 0', '#2a78d6', '-', 2, EXP/'main_hard/eval_fisher_k8/klkeep.json'),
    ('Fisher baseline (5k)', '#1baf7a', '--', 2, ROOT/'output/fisher_k4_5000/eval_fisher_k8/klkeep.json'),
    ('VPD (5k)', '#eda100', '--', 2, ROOT/'output/evaluations/gim_normal_vpd_5000/vpd/klkeep.json'),
    ('VPD published (400k)', '#e87ba4', '--', 2, ROOT/'output/published_vpd_vs_finished_parauto/vpd/klkeep.json'),
]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    output = Path(sys.argv[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=170, facecolor=SURFACE)
    for name, color, style, width, path in SERIES:
        curve = [r for r in json.loads(path.read_text())['curve'] if r['kept_components'] > 0]
        ax.plot([r['kept_components'] for r in curve], [r['kl'] for r in curve], color=color, ls=style, lw=width,
                marker='o', ms=4.5, markeredgecolor=SURFACE, markeredgewidth=1, label=name)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('pieces kept per token', color=INK)
    ax.set_ylabel('KL to the intact model', color=INK)
    ax.set_title('KLKeep, 5k steps (65,536 held-out positions)', color=INK, loc='left', fontsize=11)
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
