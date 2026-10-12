"""Does VPD's adversarial training loss help? The Matryoshka lambda = 1 new-budget run with and without VPD's persistent
PGD loss, against published VPD. Left: KLKeep (65,536 held-out positions, each method ranked by its own score).
Middle and right: VPD's adversarial eval (adversarial_probe.py --scope global --sequences 128: one value r in [0, 1]
per dropped piece over 128 x 512 held-out positions, sign ascent on the KL, step 0.1), KL after 0-320 ascent steps,
top 1% and 10% kept by each method's own ranking. Usage: plot_adversarial67.py OUTPUT_PNG"""
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
EXP = ROOT/'output/matryoshka67_20260923'
ADVERSARIAL = ROOT/'output/adversarial'
# (label, color, line style, KLKeep json, adversarial json); colors validated all-pairs (blue, magenta, green).
SERIES = [
    ('ours, λ = 1, new budgets', '#2a78d6', '-', EXP/'main_hard_lp1_strat_per1_pow1.5/eval_fisher_k8/klkeep.json',
     ADVERSARIAL/'matryoshka_main_hard_lp1_strat_per1_pow1.5_global_128.json'),
    ('same + VPD adversarial loss', '#d55181', '-', EXP/'main_hard_lp1_strat_per1_pow1.5_adv1/eval_fisher_k8/klkeep.json',
     ADVERSARIAL/'matryoshka_main_hard_lp1_strat_per1_pow1.5_adv1_global_128.json'),
    ('VPD, published (13.1B tokens)', '#008300', '--', ROOT/'output/published_vpd_vs_finished_parauto/vpd/klkeep.json',
     ADVERSARIAL/'vpd_published_global_128.json'),
]
STEPS = ('0', '20', '40', '80', '160', '320')
FRACTIONS = (0.01, 0.1)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'


def load(path, positions):
    result = json.loads(Path(path).read_text())
    assert result['positions'] == positions and result['data_sha256'] == DATA_SHA, path
    return result


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    output = Path(sys.argv[1])
    loaded = [(name, color, ls, load(klkeep, 65536), load(adversarial, 65536)) for name, color, ls, klkeep, adversarial in SERIES]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=170, facecolor=SURFACE)
    ax = axes[0]
    for name, color, ls, klkeep, _ in loaded:
        curve = [r for r in klkeep['curve'] if r['kept_components'] > 0]
        ax.plot([r['kept_components'] for r in curve], [r['kl'] for r in curve], color=color, ls=ls, lw=2.2,
                marker='o', ms=3.4, markeredgewidth=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.set_xlabel('pieces kept per token', color=INK, fontsize=9)
    ax.set_ylabel('KL to the intact model', color=INK, fontsize=9)
    ax.set_title('KLKeep (dropped pieces off)', color=INK, fontsize=10, loc='left')
    style(ax)
    for ax, fraction in zip(axes[1:], FRACTIONS):
        key = f'top{fraction:g}'
        for name, color, ls, _, adversarial in loaded:
            ax.plot(range(len(STEPS)), [adversarial['kl'][key][s] for s in STEPS], color=color, ls=ls, lw=2.2,
                    marker='o', ms=3.4, markeredgewidth=0)
        ax.set_yscale('log')
        ax.set_xticks(range(len(STEPS)), STEPS)
        ax.set_xlabel('adversary ascent steps', color=INK, fontsize=9)
        ax.set_title(f'adversarial masks, top {fraction*100:g}% kept', color=INK, fontsize=10, loc='left')
        style(ax)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, marker='o', ms=3.4, markeredgewidth=0, label=n) for n, c, ls, _, _ in loaded]
    fig.suptitle('VPD\'s adversarial training loss on our method (5k steps; 65,536 held-out positions)', color=INK, fontsize=11,
                 x=0.01, y=0.99, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for name, _, _, klkeep, adversarial in loaded:
        kept = {r['kept_components']: round(r['kl'], 3) for r in klkeep['curve']}
        print(name, 'KLKeep', kept)
        print(name, 'adversarial', {k: {s: round(v, 2) for s, v in d.items()} for k, d in adversarial['kl'].items()})


if __name__ == '__main__': main()
