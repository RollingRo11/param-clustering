"""Explanation aggregation (aggregation67.py): KL rise over the recipient's own explanation against the number of donor
tokens aggregated, real donors vs the uniform and frequency-matched random controls, one panel per explanation
(ours at two explanation sizes, VPD). Hollow markers: the audit's published VPD numbers (its own 1,024 texts).
Usage: plot_aggregation67.py OUTPUT_PNG PARAUTO_JSON VPD_JSON"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

KINDS = [('real', 'real donor tokens', '#2a78d6', '-'), ('uniform', 'random pieces, uniform', '#eb6834', ':'),
         ('frequency', 'random pieces, frequency-matched', '#1baf7a', '--')]       # slots 1-3, validated all-pairs
# The audit's Table (published VPD, its texts): rise at 1 / 8 / 64 donor tokens.
AUDIT = dict(real={1: .0038, 8: .0402, 64: .458}, uniform={1: -.0021, 8: -.0110, 64: .040}, frequency={1: -.0033, 8: -.0038})
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def ticks(x, _=None):
    return {512: '1 text', 32768: '64 texts', 524288: 'all texts'}.get(int(x), f'{int(x)}')


def main():
    output, ours, vpd = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[3]).read_text())
    panels = [(f"ours, top {s['explanation_size']} pieces per token", s) for s in ours['results'].values()]
    panels += [(f"VPD published, labels > 0.1 (~{vpd['results']['vpd_labels']['explanation_size']:.0f} per token)", vpd['results']['vpd_labels'])]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2*len(panels), 4.6), dpi=170, facecolor=SURFACE)
    for ax, (title, s) in zip(axes, panels):
        for kind, _, color, ls in KINDS:
            rows = sorted([r for r in s['rows'] if r['kind'] == kind], key=lambda r: r['donor_tokens'])
            x = [r['donor_tokens'] for r in rows]
            ax.plot(x, [r['rise']['mean'] for r in rows], color=color, ls=ls, lw=2.2, marker='o', ms=3.5, markeredgewidth=0)
            ax.fill_between(x, [r['rise']['lo'] for r in rows], [r['rise']['hi'] for r in rows], color=color, alpha=.15, lw=0)
            if title.startswith('VPD') and kind in AUDIT:
                ax.plot(list(AUDIT[kind]), list(AUDIT[kind].values()), ls='none', marker='o', ms=7, markerfacecolor='none',
                        markeredgecolor=color, markeredgewidth=1.5)
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 512, 32768, 524288]))
        ax.xaxis.set_major_formatter(FuncFormatter(ticks)); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('donor tokens aggregated', color=INK, fontsize=9)
        ax.set_title(f"{title}\nown explanation: KL {s['own_kl']['mean']:.2f}", color=INK, fontsize=9.5, loc='left')
        ax.set_facecolor(SURFACE); ax.grid(True, which='major', color=GRID, lw=.6); ax.tick_params(colors=MUTED, labelsize=8)
        for side in ('top', 'right'): ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    axes[0].set_ylabel('KL rise over the own explanation (nats)', color=INK, fontsize=9)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.2, label=n) for _, n, c, ls in KINDS]
    handles.append(Line2D([], [], ls='none', marker='o', ms=7, markerfacecolor='none', markeredgecolor=MUTED, label="audit's published VPD numbers"))
    fig.suptitle('Do explanations aggregate? Switching on other tokens\' explanations at every recipient token '
                 f"({ours['recipients'][1] - ours['recipients'][0]:,} recipient texts, 8 draws, 95% intervals)",
                 color=INK, fontsize=11, x=.01, y=.99, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=4, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0., .935))
    fig.tight_layout(rect=(0, 0, 1, .86))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)


if __name__ == '__main__': main()
