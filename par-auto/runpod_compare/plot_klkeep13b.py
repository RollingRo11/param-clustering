"""KLKeep of the Matryoshka 13B run (KISSKI, VPD's token stream, mass penalty 10) at one snapshot, against the 5k-step
Matryoshka runs (with and without the mass penalty), VPD at 5k steps (same data as the 5k runs) and published VPD
(400k steps, the same 13.1B-token stream the 13B run trains on). All curves are evaluate_checkpoint.py on the same 65,536
held-out positions (data sha e564...), each method ranked by its own score (ours: Fisher K=8; VPD: its CI).
Usage: plot_klkeep13b.py KLKEEP_DIR STEP OUTPUT_PNG   (KLKEEP_DIR holds the klkeep.json files named as in SERIES)"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def series(step):
    # (label, color, line style, file, training steps); validated palette slots as in plot_token_curve13b.py: orange = the 13B
    # run, blue = the 5k no-penalty run, aqua = the 5k mass-10 run, yellow/green = VPD (dashed).
    tokens = step * 131_072 / 1e9
    when = 'final' if step == 100_000 else f'step {step:,}'
    return [(f'ours, 13B run, {when} ({tokens:.1f}B tokens; new budgets + mass penalty 10)', '#eb6834', '-', f'm13b_{step:06d}.json', step),
            ('ours, 5k steps, new budgets + mass penalty 10 (0.16B tokens)', '#1baf7a', '-', 'mass10_5000.json', 5000),
            ('ours, 5k steps, new budgets, no mass penalty (0.16B tokens)', '#2a78d6', '-', 'nopen_5000.json', 5000),
            ('VPD, 5k steps (0.16B tokens)', '#eda100', '--', 'vpd_5000.json', 5000),
            ('VPD, published (400k steps, 13.1B tokens)', '#008300', '--', 'vpd_published.json', 400_000)]


def main():
    directory, step, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    loaded = []
    for name, color, ls, file, steps in series(step):
        result = json.loads((directory/file).read_text())
        assert result['positions'] == 65536 and result['data_sha256'] == DATA_SHA and result['training_steps'] == steps, file
        curve = [r for r in result['curve'] if r['kept_components'] > 0]
        loaded.append((name, color, ls, result, curve))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=170, facecolor=SURFACE)
    for name, color, ls, result, curve in loaded:
        ax.plot([r['kept_components'] for r in curve], [r['kl'] for r in curve], color=color, ls=ls, lw=2.2, marker='o', ms=3.6,
                markeredgewidth=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('pieces kept per token (top-k by each method\'s own score)', color=INK, fontsize=9)
    ax.set_ylabel('KL to the real model', color=INK, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, marker='o', ms=3.6, markeredgewidth=0, label=n) for n, c, ls, _, _ in loaded]
    fig.suptitle('KLKeep: Matryoshka 13B run vs 5k-step runs and VPD (65,536 held-out tokens)', color=INK,
                 fontsize=11, x=0.01, y=1.0, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=1, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for name, _, _, result, curve in loaded:
        print(name, [(int(r['kept_components']), round(r['kl'], 3)) for r in curve])


if __name__ == '__main__': main()
