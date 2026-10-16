"""Adversarial KL against training length, for every run with VPD's adversarial eval at more than one training length
(plus the older single-length par-auto runs as lone points). VPD's eval (adversarial_probe.py --scope global
--sequences 128: one value r in [0, 1] per dropped piece over 128 x 512 held-out positions, random start, sign ascent on
the KL, step 0.1), soft adversary. Rows: KL after 20 and 320 ascent steps; columns: top 1% and 10% kept by each run's own
ranking (ours: Fisher score; VPD: its CI). x = training tokens (steps differ in tokens: the 13B runs use 4 GPUs).
Usage: plot_adversarial_vs_tokens67.py ADVERSARIAL_DIR OUTPUT_PNG"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (label, color, line style, marker, adversarial json names, in training order); colors as in plot_adversarial13b.py.
SERIES = [('VPD (5k, 18.7k, 400k steps)', '#008300', '--', 'o',
           ['vpd_5000_global_128.json', 'vpd_paired_global_128.json', 'vpd_published_global_128.json']),
          ('ours, Matryoshka + mass penalty 10 (5k, 100k steps)', '#eb6834', '-', 'o',
           ['matryoshka_main_hard_lp1_strat_per1_pow1.5_mass10_global_128.json', 'matryoshka13b_100000_global_128.json']),
          ('ours, Matryoshka + VPD training attack (adv1, 5k)', '#2a78d6', '', 's',
           ['matryoshka_main_hard_lp1_strat_per1_pow1.5_adv1_global_128.json']),
          ('old par-auto, no Matryoshka (30.5k steps)', '#8a5cd1', '', 'D', ['parauto_1b_global_128.json'])]
ATTACK = ('20', '320')
FRACTIONS = (0.01, 0.1)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    directory, output = Path(sys.argv[1]), Path(sys.argv[2])
    loaded = []
    for name, color, ls, marker, files in SERIES:
        results = [json.loads((directory/f).read_text()) for f in files]
        for r in results: assert r['scope'] == 'global' and r['sequences'] == 128 and r.get('adversary', 'soft') == 'soft'
        loaded.append((name, color, ls, marker, [r['training_positions'] for r in results], [r['kl'] for r in results]))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), dpi=170, facecolor=SURFACE, sharex=True, sharey=True)
    for row, steps in enumerate(ATTACK):
        for col, fraction in enumerate(FRACTIONS):
            ax = axes[row, col]
            for name, color, ls, marker, tokens, kls in loaded:
                ax.plot(tokens, [kl[f'top{fraction:g}'][steps] for kl in kls], color=color, ls=ls or 'none', lw=2.2,
                        marker=marker, ms=6, markeredgewidth=0)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_title(f'top {fraction*100:g}% kept, {steps} attack steps', color=INK, fontsize=10, loc='left')
            ax.set_facecolor(SURFACE)
            ax.grid(True, which='major', color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ('top', 'right'): ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    for ax in axes[:, 0]: ax.set_ylabel('KL to the real model', color=INK, fontsize=9)
    for ax in axes[1]: ax.set_xlabel('training tokens', color=INK, fontsize=9)
    handles = [Line2D([], [], color=c, ls=ls or 'none', lw=2.2, marker=m, ms=6, markeredgewidth=0, label=n)
               for n, c, ls, m, _, _ in loaded]
    fig.suptitle('Adversarial masks vs training length', color=INK, fontsize=11, x=0.01, y=1.0, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)


if __name__ == '__main__': main()
