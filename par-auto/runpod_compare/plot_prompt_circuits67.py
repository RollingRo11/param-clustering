"""Description length of prompt circuits (prompt_circuits67.py): VPD's per-prompt optimization with pieces outside the
circuit off and the KL to the intact next-token distribution at the last position as the loss, for published VPD and our
13B run, on the princess prompt and 32 held-out 10-token prompts. Left and middle: per prompt, the fewest nodes among the
sparsity sweep's circuits whose KL (circuit pieces at full strength, everything else off) is at most 0.1 / 0.5, ours
against VPD (points on the top or right edge: that method never got there). Right: every circuit of the sweep, nodes
against that KL; lines join the per-coefficient medians. Usage: plot_prompt_circuits67.py OUTPUT_PNG SHARD_JSON..."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

COLORS = dict(vpd='#008300', ours='#eb6834')
NAMES = dict(vpd='VPD, published', ours='ours, 13B run')
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(True, which='major', color=GRID, lw=0.6); ax.tick_params(colors=MUTED, labelsize=8)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)


def main():
    output, shards = Path(sys.argv[1]), [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    prompts = {k: v for s in shards for k, v in s['prompts'].items() if len(v.get('ours', [])) == len(s['sweep'])}
    sweep, thresholds = shards[0]['sweep'], shards[0]['thresholds']
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=170, facecolor=SURFACE)
    summary = {}
    for ax, t in zip(axes[:2], thresholds):
        pairs = [(p['vpd_description_length'][str(t)], p['ours_description_length'][str(t)]) for p in prompts.values()]
        finite = [v for pair in pairs for v in pair if v is not None]
        edge = max(finite) * 2 if finite else 1000
        xs = [v if v is not None else edge for v, _ in pairs]; ys = [o if o is not None else edge for _, o in pairs]
        ax.scatter(xs, ys, s=22, color=MUTED, alpha=0.75, lw=0)
        both = [(v, o) for v, o in pairs if v is not None and o is not None]
        lo = min(min(xs), min(ys)) / 1.5
        ax.plot([lo, edge * 1.2], [lo, edge * 1.2], color=GRID, lw=1, zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, edge * 1.3); ax.set_ylim(lo, edge * 1.3)
        for axis in (ax.xaxis, ax.yaxis): axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('VPD: fewest nodes', color=INK, fontsize=9); ax.set_ylabel('ours: fewest nodes', color=INK, fontsize=9)
        smaller = sum(o < v for v, o in both); larger = sum(o > v for v, o in both)
        ax.set_title(f'KL <= {t:g}: ours smaller on {smaller}, larger on {larger} of {len(pairs)} prompts', color=INK, fontsize=9.5, loc='left')
        style(ax)
        summary[t] = dict(prompts=len(pairs), vpd_reached=sum(v is not None for v, _ in pairs), ours_reached=sum(o is not None for _, o in pairs),
                          ours_smaller=smaller, ours_larger=larger, ties=len(both) - smaller - larger,
                          median_ratio_ours_over_vpd=float(np.median([o / v for v, o in both])) if both else None)
    ax = axes[2]
    for m in ('vpd', 'ours'):
        runs = [r for p in prompts.values() for r in p[m]]
        ax.scatter([r['nodes'] for r in runs], [max(r['kl_binary'], 1e-4) for r in runs], s=8, color=COLORS[m], alpha=0.25, lw=0)
        med = [(np.median([r['nodes'] for p in prompts.values() for r in p[m] if r['imp_min_coeff'] == c]),
                np.median([r['kl_binary'] for p in prompts.values() for r in p[m] if r['imp_min_coeff'] == c])) for c in sweep]
        ax.plot([a for a, _ in med], [b for _, b in med], color=COLORS[m], lw=2.2, marker='o', ms=4, markeredgewidth=0)
    for t in thresholds: ax.axhline(t, color=MUTED, lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    for axis in (ax.xaxis, ax.yaxis): axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('nodes in the circuit', color=INK, fontsize=9); ax.set_ylabel('KL to the intact model (only the circuit, rest off)', color=INK, fontsize=9)
    ax.set_title('every circuit of the sweep (lines: medians per coefficient)', color=INK, fontsize=9.5, loc='left'); style(ax)
    handles = [Line2D([], [], color=COLORS[m], lw=2.2, marker='o', ms=4, label=NAMES[m]) for m in ('vpd', 'ours')]
    fig.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.95))
    fig.suptitle(f'Circuit size at matched faithfulness: pieces outside the circuit off, KL loss at the last position ({len(prompts)} prompts)',
                 color=INK, fontsize=11, x=0.01, y=1.0, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, json.dumps(summary), flush=True)


if __name__ == '__main__': main()
