"""Emoticon circuits and the VPD paper's emoticon edit (emoticon67.py) with the finished Matryoshka 13B run (step 100,000,
13.1B tokens of VPD's stream, new budgets + mass penalty 10) next to ours at 5k steps (mass penalty 10), VPD at 5k steps
and published VPD. Same panels as plot_emoticon67.py: left, median KL at the 512 emoticon starters against the pieces
each token keeps (hollow markers: VPD's own explanations, label > 0.1, at their median size); right, the edit of each
model's most emoticon-selective residual-writing piece at alpha = 1..128, mean P(o) at the starters against the KL on
emoticon-free text. Colors as in plot_klkeep13b.py. Usage: plot_emoticon13b.py OUTPUT_PNG FOLDER"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# Validated palette slots as in plot_klkeep13b.py: orange = the 13B run, aqua = ours 5k (mass 10), yellow/green = VPD.
SERIES = [('ours, 13B run, final (13.1B tokens)', 'ours_13b_final.json', '#eb6834', '-'),
          ('ours, 5k steps, mass penalty 10', 'ours_mass10_5000.json', '#1baf7a', '-'),
          ('VPD, 5k steps', 'vpd_5000.json', '#eda100', '--'),
          ('VPD, published (400k steps)', 'vpd_published.json', '#008300', '--')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which='major', color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)


def main():
    output, folder = Path(sys.argv[1]), Path(sys.argv[2])
    runs = [(name, json.loads((folder/f).read_text()), color, ls) for name, f, color, ls in SERIES]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=170, facecolor=SURFACE)
    ax = axes[0]
    for name, d, color, ls in runs:
        ks = sorted(d['kl_at_starter'], key=int)
        ax.plot([int(k) for k in ks], [d['kl_at_starter'][k]['median'] for k in ks], color=color, ls=ls, lw=2.2, marker='o', ms=3.4,
                markeredgewidth=0)
        own = d.get('own_explanation')
        if own: ax.plot([own['size_median']], [own['kl_median']], ls='none', marker='o', ms=9, markerfacecolor='none', markeredgecolor=color,
                        markeredgewidth=1.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 5.)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('pieces kept per token', color=INK, fontsize=9)
    ax.set_ylabel('KL at the emoticon starter (median)', color=INK, fontsize=9)
    ax.set_title('circuit size at 512 emoticon starters', color=INK, fontsize=10, loc='left')
    style(ax)
    ax = axes[1]
    for name, d, color, ls in runs:
        alphas = sorted((a for a in d['edits'] if a != '0'), key=float)
        ax.plot([d['edits'][a]['kl_global'] for a in alphas], [d['edits'][a]['p_o'] for a in alphas], color=color, ls=ls, lw=2.2,
                marker='o', ms=3.6, markeredgewidth=0)
        first = d['edits'][alphas[0]]
        ax.annotate('α = 1', (first['kl_global'], first['p_o']), textcoords='offset points', xytext=(6, -10), fontsize=7.5, color=MUTED)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.,)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('KL on emoticon-free text (off-target damage)', color=INK, fontsize=9)
    ax.set_ylabel('P(o) after the emoticon starter', color=INK, fontsize=9)
    ax.set_title('editing one piece to predict :o', color=INK, fontsize=10, loc='left')
    style(ax)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, label=n) for n, _, c, ls in runs]
    handles.append(Line2D([], [], ls='none', marker='o', ms=8, markerfacecolor='none', markeredgecolor=MUTED, label="VPD's own explanation (label > 0.1)"))
    fig.suptitle('Emoticon circuits and the :o edit (512 emoticons from unseen training text)', color=INK, fontsize=11, x=0.01, y=0.99, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.945))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)


if __name__ == '__main__': main()
