"""Does the type of adversary matter? VPD's adversarial eval (adversarial_probe.py --scope global --sequences 128: one
value r per dropped piece over 128 x 512 held-out positions, sign ascent on the KL, step 0.1) with the soft adversary
(dropped piece scaled by r) against the hard one (dropped piece fully on where r > 0.5, straight-through gradient),
started at random r or with every dropped piece off. Our lambda = 1 new-budget run (hard forward) and published VPD, top
1% and 10% kept by each method's own ranking. Usage: plot_adversarial_hard67.py OUTPUT_PNG"""
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
ADVERSARIAL = ROOT/'output/adversarial'
# Color = method (the 2026-10-05 figure's blue and green), line style = adversary.
METHODS = [('ours, λ = 1, new budgets', '#2a78d6', 'matryoshka_main_hard_lp1_strat_per1_pow1.5_global_128'),
           ('VPD, published', '#008300', 'vpd_published_global_128')]
ADVERSARIES = [('soft adversary (r scales the piece)', '-', ''), ('hard adversary, random start', '--', '_hard_random'),
               ('hard adversary, every dropped piece off at start', ':', '_hard_off')]
STEPS = ('0', '20', '40', '80', '160', '320')
FRACTIONS = (0.01, 0.1)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'


def load(stem, suffix):
    result = json.loads((ADVERSARIAL/f'{stem}{suffix}.json').read_text())
    assert result['positions'] == 65536 and result['data_sha256'] == DATA_SHA and result['scope'] == 'global', stem
    assert result.get('adversary', 'soft') == ('hard' if suffix else 'soft'), stem + suffix
    assert result.get('init', 'random') == ('off' if suffix.endswith('off') else 'random'), stem + suffix
    return result


def main():
    output = Path(sys.argv[1])
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(10.5, 4.3), dpi=170, facecolor=SURFACE)
    for ax, fraction in zip(axes, FRACTIONS):
        key = f'top{fraction:g}'
        for name, color, stem in METHODS:
            for kind, ls, suffix in ADVERSARIES:
                kl = load(stem, suffix)['kl'][key]
                ax.plot(range(len(STEPS)), [kl[s] for s in STEPS], color=color, ls=ls, lw=2.2, marker='o', ms=3.4,
                        markeredgewidth=0)
                print(name, '|', kind, '|', key, {s: round(kl[s], 2) for s in STEPS})
        ax.set_yscale('log')
        ax.set_xticks(range(len(STEPS)), STEPS)
        ax.set_xlabel('adversary ascent steps', color=INK, fontsize=9)
        ax.set_title(f'top {fraction*100:g}% kept', color=INK, fontsize=10, loc='left')
        ax.set_facecolor(SURFACE); ax.grid(True, which='major', color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=8)
        for side in ('top', 'right'): ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel('KL to the intact model', color=INK, fontsize=9)
    handles = [Line2D([], [], color=c, lw=2.2, label=n) for n, c, _ in METHODS]
    handles += [Line2D([], [], color=MUTED, ls=ls, lw=2.2, label=k) for k, ls, _ in ADVERSARIES]
    fig.suptitle('Soft vs hard adversary (128 x 512 held-out positions)', color=INK, fontsize=11, x=0.01, y=0.99, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)


if __name__ == '__main__': main()
