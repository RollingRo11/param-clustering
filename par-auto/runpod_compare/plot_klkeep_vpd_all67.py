"""KLKeep for slides: the Matryoshka lambda = 1 new-budget run (5k steps) against every VPD checkpoint we have KLKeep
for (the paired run at 1k, 2k, 5k, 10k, 15k steps and its 18,664-step end, and the published 400k-step run). Same
65,536 held-out positions; each method ranked by its own score (ours Fisher K=8, VPD its causal importance).
Usage: plot_klkeep_vpd_all67.py OUTPUT_PNG"""
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
OURS = ROOT/'output/matryoshka67_20260923/main_hard_lp1_strat_per1_pow1.5/eval_fisher_k8/klkeep.json'
# VPD checkpoints in training order; a one-hue green ramp, light -> dark with tokens (validated --ordinal).
VPD = [ROOT/'output'/p/'klkeep.json' for p in ('evaluations/gim_vpd_1000/vpd', 'vpd', 'evaluations/gim_normal_vpd_5000/vpd',
                                               'evaluations/10000/vpd', 'evaluations/15000/vpd', 'ce_curve_20260916/vpd_0.6b',
                                               'published_vpd_vs_finished_parauto/vpd')]
GREENS = ['#68c36d', '#51ad57', '#3a9742', '#1f812d', '#006c15', '#045514', '#004100']
BLUE = '#2a78d6'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'


def load(path):
    result = json.loads(Path(path).read_text())
    assert result['positions'] == 65536 and result['data_sha256'] == DATA_SHA, path
    return result


def tokens(n):
    return f'{n/1e9:.1f}B' if n >= 1e9 else f'{n/1e6:.0f}M'


def main():
    output = Path(sys.argv[1])
    ours, vpd = load(OURS), [load(p) for p in VPD]
    assert [v['method'] for v in vpd] == ['vpd']*len(vpd) and vpd == sorted(vpd, key=lambda v: v['training_positions'])
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=200, facecolor=SURFACE)
    for result, color in zip(vpd, GREENS):
        curve = [r for r in result['curve'] if r['kept_components'] > 0]
        published = result['training_steps'] == 400000
        label = (f"VPD published: {tokens(result['training_positions'])} tokens ({result['training_steps']:,} steps)" if published
                 else f"VPD: {tokens(result['training_positions'])} tokens ({result['training_steps']:,} steps)")
        ax.plot([r['kept_components'] for r in curve], [r['kl'] for r in curve], color=color, ls='--', lw=2.4 if published else 1.7,
                marker='o', ms=3, markeredgewidth=0, label=label)
    curve = [r for r in ours['curve'] if r['kept_components'] > 0]
    ax.plot([r['kept_components'] for r in curve], [r['kl'] for r in curve], color=BLUE, lw=3, marker='o', ms=4, markeredgewidth=0,
            label=f"ours, λ = 1, new budgets: {tokens(ours['training_positions'])} tokens ({ours['training_steps']:,} steps)", zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}')); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('pieces kept per token', color=INK, fontsize=11)
    ax.set_ylabel('KL to the intact model', color=INK, fontsize=11)
    ax.set_title('KLKeep: our new-budget run vs every VPD checkpoint (65,536 held-out positions)', color=INK, fontsize=12, loc='left')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9, labelcolor=INK, loc='lower left')
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=10)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for name, result in [('ours', ours)] + [(f"vpd {tokens(v['training_positions'])}", v) for v in vpd]:
        print(name, {f"{r['keep_fraction']*100:g}%": (int(r['kept_components']), round(r['kl'], 3)) for r in result['curve']
                     if r['keep_fraction'] in (0.0003, 0.001, 0.01, 0.03, 0.1, 0.2)})


if __name__ == '__main__': main()
