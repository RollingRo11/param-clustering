"""KLKeep against training tokens: the two Matryoshka runs (lambda = 1, and lambda = 1 with the new budgets) at every
saved checkpoint, the paired VPD run at its checkpoints, and the published VPD run. Same 65,536 held-out positions,
each method ranked by its own score; one panel per keep fraction. Usage: plot_token_curve67.py OUTPUT_PNG"""
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
EXP = ROOT/'output/matryoshka67_20260923'
STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)
OURS = [('ours, λ = 1', '#4a3aa7', EXP/'main_hard_lp1'),                              # violet, as in the earlier figures
        ('ours, λ = 1, new budgets', '#2a78d6', EXP/'main_hard_lp1_strat_per1_pow1.5')]  # blue (validated all-pairs)
VPD_COLOR = '#008300'                                                                    # green, as "VPD paired" before
VPD_PAIRED = [ROOT/'output'/p/'klkeep.json' for p in ('evaluations/gim_vpd_1000/vpd', 'vpd', 'evaluations/gim_normal_vpd_5000/vpd',
                                                      'evaluations/10000/vpd', 'evaluations/15000/vpd', 'ce_curve_20260916/vpd_0.6b')]
VPD_PUBLISHED = ROOT/'output/published_vpd_vs_finished_parauto/vpd/klkeep.json'
FRACTIONS = (0.001, 0.01, 0.1)
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'


def load(path):
    result = json.loads(Path(path).read_text())
    assert result['positions'] == 65536 and result['data_sha256'] == DATA_SHA, path
    return result


def at(result, fraction):
    """(tokens trained, kept pieces, KL) at one keep fraction."""
    row, = [r for r in result['curve'] if abs(r['keep_fraction'] - fraction) < 1e-9]
    return result['training_positions'], int(row['kept_components']), row['kl']


def tokens_label(x, _=None):
    return f'{x/1e9:g}B' if x >= 1e9 else f'{x/1e6:g}M'


def main():
    output = Path(sys.argv[1])
    ours = [(name, color, [load(run/'eval_fisher_k8_steps'/str(s)/'klkeep.json') for s in STEPS] + [load(run/'eval_fisher_k8/klkeep.json')])
            for name, color, run in OURS]
    paired, published = [load(p) for p in VPD_PAIRED], load(VPD_PUBLISHED)
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(12.4, 4.3), dpi=170, facecolor=SURFACE)
    for ax, fraction in zip(axes, FRACTIONS):
        for name, color, results in ours:
            points = [at(r, fraction) for r in results]
            ax.plot([p[0] for p in points], [p[2] for p in points], color=color, lw=2.2, marker='o', ms=3.2, markeredgewidth=0)
        points = [at(r, fraction) for r in paired]
        ax.plot([p[0] for p in points], [p[2] for p in points], color=VPD_COLOR, lw=2, ls='--', marker='o', ms=3.6, markeredgewidth=0)
        x, _, kl = at(published, fraction)
        ax.plot([x], [kl], ls='none', marker='o', ms=7, markerfacecolor=SURFACE, markeredgecolor=VPD_COLOR, markeredgewidth=2)
        ours_k, vpd_k = at(ours[0][2][-1], fraction)[1], at(published, fraction)[1]
        ax.set_title(f'{fraction*100:g}% kept  ({ours_k:,} pieces; VPD {vpd_k:,})', color=INK, fontsize=10, loc='left')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1.2e7, 2.5e10)
        ax.xaxis.set_major_locator(FixedLocator([1e7, 1e8, 1e9, 1e10]))
        ax.xaxis.set_major_formatter(tokens_label); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 3., 5.)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('tokens trained', color=INK, fontsize=9)
        ax.set_facecolor(SURFACE)
        ax.grid(True, which='major', color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ('top', 'right'): ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    axes[0].set_ylabel('KL to the intact model', color=INK, fontsize=9)
    handles = [Line2D([], [], color=c, lw=2.2, marker='o', ms=3.2, markeredgewidth=0, label=n) for n, c, _ in ours] + [
        Line2D([], [], color=VPD_COLOR, lw=2, ls='--', marker='o', ms=3.6, markeredgewidth=0, label='VPD, paired run (checkpoints of one 1B-token run)'),
        Line2D([], [], ls='none', marker='o', ms=7, markerfacecolor=SURFACE, markeredgecolor=VPD_COLOR, markeredgewidth=2,
               label='VPD, published run (13.1B tokens)')]
    fig.suptitle('KLKeep vs tokens trained (65,536 held-out positions)', color=INK, fontsize=11, x=0.01, y=0.99, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=4, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for name, _, results in ours + [('VPD paired', None, paired), ('VPD published', None, [published])]:
        print(name, [(tokens_label(r['training_positions']), [round(at(r, f)[2], 3) for f in FRACTIONS]) for r in results])


if __name__ == '__main__': main()
