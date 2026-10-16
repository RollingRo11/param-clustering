"""MAttr prompt circuits in VPD's regime (prompt_mattr67.py --regime vpd), published VPD against our 13B run. Same panels
as plot_prompt_mattr67.py, one row per way of setting the pieces outside the circuit: random levels (top) and
adversarial levels (bottom). Left and middle: per prompt, the description length (the fewest pieces, top-k of the learned
ranking over (position, piece) pairs, whose KL to the intact next-token distribution at the last position is at most
0.1 / 0.5), ours against VPD; points on the top or right edge never got there. Right: KL against k, median over prompts,
for the MAttr ranking (solid) and the gradient-attribution ranking it starts from (dashed).
Usage: plot_prompt_mattr_vpd67.py OUTPUT_PNG SHARD_JSON..."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

COLORS = dict(vpd='#008300', ours='#eb6834')
NAMES = dict(vpd='VPD, published', ours='ours, 13B run')
ROWS = dict(kl_random='Pieces outside the circuit at random levels (VPD delta at a random level too)',
            kl_adversarial='Pieces outside the circuit at adversarial levels (PGD, 20 steps of 0.1, one level per piece for all positions)')
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
fmt = FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v >= 1000 else f'{v:g}')


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which='major', color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)


def median_curve(prompts, m, ranking, key, grid):
    """Median over prompts of the KL on the method's grid; a prompt with fewer pairs keeps its all-on value beyond them."""
    rows = []
    for p in prompts.values():
        pts = {pt['k']: pt[key] for pt in p[m]['curves'][ranking]}
        top = max(pts)
        rows.append([pts[min(k, top)] for k in grid])
    return np.median(np.array(rows), 0)


def main():
    output, shards = Path(sys.argv[1]), [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    prompts = {k: v for s in shards for k, v in s['prompts'].items()}
    thresholds, meta = shards[0]['thresholds'], shards[0]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), dpi=170, facecolor=SURFACE)
    summary = {}
    for row, (key, row_name) in zip(axes, ROWS.items()):
        for ax, t in zip(row[:2], thresholds):
            pairs = [(p['vpd']['description_length']['mattr'][key][str(t)], p['ours']['description_length']['mattr'][key][str(t)]) for p in prompts.values()]
            reached = [v for pair in pairs for v in pair if v is not None]
            top = max(max(meta['vpd']['grid']), max(meta['ours']['grid']))
            edge = top * 2 if len(reached) < 2 * len(pairs) else max(reached) * 1.6     # the never-reached edge, if needed
            xs = [v if v is not None else edge for v, _ in pairs]; ys = [o if o is not None else edge for _, o in pairs]
            ax.scatter(xs, ys, s=14, color=MUTED, alpha=0.6, lw=0)
            ax.plot([0.8, edge * 1.2], [0.8, edge * 1.2], color=GRID, lw=1, zorder=0)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.8, edge * 1.3); ax.set_ylim(0.8, edge * 1.3)
            for axis in (ax.xaxis, ax.yaxis): axis.set_major_formatter(fmt); axis.set_minor_formatter(NullFormatter())
            ax.set_xlabel('VPD: fewest pieces', color=INK, fontsize=9); ax.set_ylabel('ours: fewest pieces', color=INK, fontsize=9)
            both = [(v, o) for v, o in pairs if v is not None and o is not None]
            smaller, larger = sum(o < v for v, o in both), sum(o > v for v, o in both)
            ax.set_title(f'KL <= {t:g}: ours fewer on {smaller}, more on {larger} of {len(pairs)}', color=INK, fontsize=9.5, loc='left')
            style(ax)
            summary[f'{key} {t}'] = dict(prompts=len(pairs), vpd_reached=sum(v is not None for v, _ in pairs), ours_reached=sum(o is not None for _, o in pairs),
                                         ours_fewer=smaller, ours_more=larger, ties=len(both) - smaller - larger,
                                         median_vpd=float(np.median([v for v, _ in both])) if both else None,
                                         median_ours=float(np.median([o for _, o in both])) if both else None,
                                         median_ratio_ours_over_vpd=float(np.median([o / v for v, o in both])) if both else None)
        ax = row[2]
        for m in ('vpd', 'ours'):
            grid = meta[m]['grid']
            for ranking, ls in (('mattr', '-'), ('gradient', '--')):
                ax.plot(grid, median_curve(prompts, m, ranking, key, grid), color=COLORS[m], ls=ls, lw=2 if ls == '-' else 1.4)
        for t in thresholds: ax.axhline(t, color=MUTED, lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        for axis in (ax.xaxis, ax.yaxis): axis.set_major_formatter(fmt)
        ax.set_xlabel('pieces kept (top-k over all positions)', color=INK, fontsize=9)
        ax.set_ylabel('KL to the intact model at the last position', color=INK, fontsize=9)
        ax.set_title('median over prompts', color=INK, fontsize=9.5, loc='left'); style(ax)
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=NAMES[m]) for m in ('vpd', 'ours')] + [
        Line2D([], [], color=MUTED, lw=2, label='MAttr ranking'), Line2D([], [], color=MUTED, lw=1.4, ls='--', label='gradient-attribution ranking')]
    fig.legend(handles=handles, loc='upper left', ncol=4, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.975))
    fig.suptitle(f'Prompt circuits by MAttr in VPD\'s regime: pieces outside the circuit at random or adversarial levels ({len(prompts)} prompts)',
                 color=INK, fontsize=11, x=0.01, y=1.0, ha='left')
    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.06, right=0.99, hspace=0.5, wspace=0.28)
    for row, row_name in zip(axes, ROWS.values()):
        fig.text(0.01, row[0].get_position().y1 + 0.045, row_name, color=INK, fontsize=10.5, fontweight='bold', ha='left')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, json.dumps(summary), flush=True)


if __name__ == '__main__': main()
