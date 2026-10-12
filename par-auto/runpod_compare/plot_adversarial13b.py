"""Adversarial eval of the finished Matryoshka 13B run (step 100,000, 13.1B tokens of VPD's stream, new budgets + mass
penalty 10) against the 5k-step Matryoshka runs (new budgets, with and without the mass penalty), VPD at 5k steps and
published VPD. VPD's adversarial eval (adversarial_probe.py --scope global --sequences 128, code snapshot
adversarial67_hard_20261006: one value r in [0, 1] per dropped piece over 128 x 512 held-out positions, random start,
sign ascent on the KL, step 0.1), KL after 0-320 ascent steps at 0.01%, 0.1%, 1% and 10% kept by each run's own ranking
(ours: Fisher score; VPD: its CI). For the older runs 1% and 10% come from the original probe files and 0.01% and 0.1%
from the *_small.json reruns (see plot_mass_adversarial_only67.py); the 13B run has all four in one file.
Usage: plot_adversarial13b.py ADVERSARIAL_DIR OUTPUT_PNG"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# (label, color, line style, adversarial json files merged in order, training steps); colors as in plot_klkeep13b.py.
SERIES = [('ours, 13B run, final (13.1B tokens; new budgets + mass penalty 10)', '#eb6834', '-',
           ['matryoshka13b_100000_global_128.json'], 100_000),
          ('ours, 5k steps, new budgets + mass penalty 10', '#1baf7a', '-',
           ['matryoshka_main_hard_lp1_strat_per1_pow1.5_mass10_global_128.json',
            'matryoshka_main_hard_lp1_strat_per1_pow1.5_mass10_global_128_small.json'], 5000),
          ('ours, 5k steps, new budgets, no mass penalty', '#2a78d6', '-',
           ['matryoshka_main_hard_lp1_strat_per1_pow1.5_global_128.json',
            'matryoshka_main_hard_lp1_strat_per1_pow1.5_global_128_small.json'], 5000),
          ('VPD, 5k steps', '#eda100', '--', ['vpd_5000_global_128.json', 'vpd_5000_global_128_small.json'], 5000),
          ('VPD, published (400k steps, 13.1B tokens)', '#008300', '--',
           ['vpd_published_global_128.json', 'vpd_published_global_128_small.json'], 400_000)]
STEPS = ('0', '20', '40', '80', '160', '320')
FRACTIONS = (0.0001, 0.001, 0.01, 0.1)
COUNTS = dict(ours=43_008, vpd=38_912)          # pieces in total; kept = round(fraction * total)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'


def main():
    directory, output = Path(sys.argv[1]), Path(sys.argv[2])
    loaded = []
    for name, color, ls, files, steps in SERIES:
        kl = {}
        for file in files:
            r = json.loads((directory/file).read_text())
            assert r['positions'] == 65536 and r['data_sha256'] == DATA_SHA and r['completed'] == steps and r['scope'] == 'global', file
            assert r.get('adversary', 'soft') == 'soft' and r.get('init', 'random') == 'random' and r['step_size'] == 0.1, file
            kl.update({k: v for k, v in r['kl'].items() if k not in kl})       # earlier files win (originals over reruns)
        assert all(f'top{f:g}' in kl for f in FRACTIONS), name
        loaded.append((name, color, ls, kl))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.4), dpi=170, facecolor=SURFACE, sharey=True)
    for ax, fraction in zip(axes.flat, FRACTIONS):
        key = f'top{fraction:g}'
        for name, color, ls, kl in loaded:
            ax.plot(range(len(STEPS)), [kl[key][s] for s in STEPS], color=color, ls=ls, lw=2.2, marker='o', ms=3.6, markeredgewidth=0)
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
    fig.suptitle('Adversarial masks: finished Matryoshka 13B run vs 5k-step runs and VPD', color=INK, fontsize=11, x=0.01, y=1.0, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.975))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for name, _, _, kl in loaded:
        print(name, {f'top{f:g}': {s: round(kl[f'top{f:g}'][s], 2) for s in STEPS} for f in FRACTIONS})


if __name__ == '__main__': main()
