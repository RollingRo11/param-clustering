"""Adversarial eval only: the Matryoshka lambda = 1 new-budget 5k run without and with the piece-mass penalty (weight 10),
against VPD at 5k steps (same data) and published VPD (400k steps, 13.1B tokens), each ranked by its own CI.
VPD's adversarial eval (adversarial_probe.py --scope global --sequences 128: one value r in [0, 1] per dropped piece over
128 x 512 held-out positions, random start, sign ascent on the KL, step 0.1), KL after 0-320 ascent steps, top 0.01%, 0.1%, 1% and 10% kept by each
run's own ranking (ours: Fisher score; VPD: its CI). 1% and 10% come from the original probe files, 0.01% and 0.1% from
the *_small.json reruns (same code snapshot; they also redo 1%, which must match). Usage: plot_mass_adversarial_only67.py ADVERSARIAL_DIR OUTPUT_PNG"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# (label, color, line style, adversarial json name, training steps); colors from plot_mass_adversarial67.py's validated
# palette (blue, orange, yellow, green), VPD dashed.
SERIES = [('ours, no mass penalty (piece mass 215%)', '#2a78d6', '-', 'matryoshka_main_hard_lp1_strat_per1_pow1.5_global_128.json', 5000),
          ('ours, mass penalty 10 (piece mass 100%)', '#eb6834', '-', 'matryoshka_main_hard_lp1_strat_per1_pow1.5_mass10_global_128.json', 5000),
          ('VPD, 5k steps (same data)', '#eda100', '--', 'vpd_5000_global_128.json', 5000),
          ('VPD, published (400k steps)', '#008300', '--', 'vpd_published_global_128.json', 400_000)]
STEPS = ('0', '20', '40', '80', '160', '320')
FRACTIONS = (0.0001, 0.001, 0.01, 0.1)
COUNTS = dict(ours=43_008, vpd=38_912)          # pieces in total; kept = round(fraction * total)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'


def main():
    directory, output = Path(sys.argv[1]), Path(sys.argv[2])
    loaded = []
    for name, color, ls, file, steps in SERIES:
        result, small = [json.loads((directory/f).read_text()) for f in (file, file.replace('.json', '_small.json'))]
        for r in (result, small):
            assert r['positions'] == 65536 and r['data_sha256'] == DATA_SHA and r['completed'] == steps and r['scope'] == 'global', file
        assert small['checkpoint_sha256'] == result['checkpoint_sha256'] and small.get('adversary', 'soft') == 'soft', file
        gap = max(abs(small['kl']['top0.01'][s] / result['kl']['top0.01'][s] - 1) for s in STEPS)
        print(name, 'rerun of 1% kept vs original, largest relative difference', f'{gap:.2e}', flush=True)
        loaded.append((name, color, ls, {**result['kl'], **{k: small['kl'][k] for k in ('top0.0001', 'top0.001')}}))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), dpi=170, facecolor=SURFACE, sharey=True)
    for ax, fraction in zip(axes.flat, FRACTIONS):
        key = f'top{fraction:g}'
        for name, color, ls, result in loaded:
            ax.plot(range(len(STEPS)), [result[key][s] for s in STEPS], color=color, ls=ls, lw=2.2, marker='o', ms=3.6,
                    markeredgewidth=0)
        ax.set_yscale('log')
        ax.set_xticks(range(len(STEPS)), STEPS)
        ax.set_xlabel('adversary steps', color=INK, fontsize=9)
        ours, vpd = (round(fraction * n) for n in COUNTS.values())
        ax.set_title(f'top {fraction*100:g}% kept ({ours:,} of our pieces, {vpd:,} of VPD\'s)', color=INK, fontsize=10, loc='left')
        ax.set_facecolor(SURFACE)
        ax.grid(True, which='major', color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ('top', 'right'): ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in axes[:, 0]: ax.set_ylabel('KL to the real model', color=INK, fontsize=9)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, marker='o', ms=3.6, markeredgewidth=0, label=n) for n, c, ls, _ in loaded]
    fig.suptitle('Adversarial masks: Matryoshka with and without the mass penalty, and VPD', color=INK, fontsize=11, x=0.01, y=1.0, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for name, _, _, result in loaded:
        print(name, {f'top{f:g}': {s: round(result[f'top{f:g}'][s], 2) for s in STEPS} for f in FRACTIONS})


if __name__ == '__main__': main()
