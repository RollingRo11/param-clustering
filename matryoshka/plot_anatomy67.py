"""Anatomy of VPD's adversarial eval (adversary_anatomy67.py) for our new-budget run without and with the piece-mass
penalty (weight 10) and published VPD. Left, middle: KL when only the N pieces with the largest first-order share of the
20-step attack keep the attack's value (all others back to 0), top 1% and 10% kept. Right: KL when the attack is
applied in one layer only, top 1% kept. Usage: plot_anatomy67.py OUTPUT_PNG ANATOMY_DIR"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

# Ours in categorical slots 1 and 3 (blue, aqua; validated all-pairs); VPD as the muted dashed reference.
SERIES = [('ours, new budgets, no penalty', 'no_penalty.json', '#2a78d6', '-'),
          ('ours, mass penalty 10', 'mass10.json', '#1baf7a', '-'),
          ('VPD, published', 'vpd_published.json', '#52514e', '--')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which='major', color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())


def main():
    output, folder = Path(sys.argv[1]), Path(sys.argv[2])
    runs = [(name, json.loads((folder/f).read_text()), color, ls) for name, f, color, ls in SERIES]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), dpi=170, facecolor=SURFACE)
    for ax, config in zip(axes[:2], ('top0.01', 'top0.1')):
        for name, d, color, ls in runs:
            top = d['attack'][config]['kl_top_n']
            ns = sorted(top, key=int)
            x = [int(n) if i < len(ns) - 1 else 1e5 for i, n in enumerate(ns)]    # the last point is every piece
            ax.plot(x, [top[n] for n in ns], color=color, ls=ls, lw=2.2, marker='o', ms=3.6, markeredgewidth=0)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000, 1e5]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: 'all' if v >= 1e5 else f'{int(v):,}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('attacked pieces switched on (largest share first)', color=INK, fontsize=9)
        ax.set_title(f'top {float(config[3:])*100:g}% kept', color=INK, fontsize=10, loc='left')
        style(ax)
    axes[0].set_ylabel('KL to the intact model', color=INK, fontsize=9)
    ax = axes[2]
    width = 0.26
    for j, (name, d, color, ls) in enumerate(runs):
        layer = d['attack']['top0.01']['kl_one_layer']
        ax.bar([L + (j - 1) * width for L in range(4)], [layer[str(L)] for L in range(4)], width=width - 0.03, color=color,
               edgecolor=SURFACE, linewidth=1.5, hatch='//' if ls == '--' else None)
    ax.set_xticks(range(4), [f'layer {L}' for L in range(4)])
    ax.set_title('attack in one layer only, top 1% kept', color=INK, fontsize=10, loc='left')
    style(ax)
    ax.set_ylim(0.2, 60)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, label=n) for n, _, c, ls in runs]
    fig.suptitle('Where the 20-step attack does its damage (VPD\'s adversarial eval, 128 x 512 held-out positions)', color=INK,
                 fontsize=11, x=0.01, y=0.99, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)


if __name__ == '__main__': main()
