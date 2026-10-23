"""KLKeep of the SimpleStories runs against VPD, one panel per ranking (per-token Fisher K=8, cached text-averaged
Fisher, write size), plus the global adversarial probe at each run's own ranking. Reads the eval JSONs that
eval_ss_run.sh writes; a run without a given eval is left out of that panel.
Usage: plot_ss_klkeep.py OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = '/root/runs'
VPD = (f'{RUNS}/old/vpd', 'VPD s-9e774a82 (CI)')
# (run dir, label, trained with, colour): categorical slots 1-3 of the dataviz reference palette, in fixed order.
TRAINED = [(f'{RUNS}/old/main_tail', 'main_tail (Fisher)', 'fisher', '#2a78d6'),
           (f'{RUNS}/matryoshka_ss_mainline/static_5k', 'static_5k (static)', 'static', '#eb6834'),
           (f'{RUNS}/matryoshka_ss_mainline/write_5k', 'write_5k @2137 (write)', 'write', '#1baf7a'),
           (f'{RUNS}/matryoshka_ss_mainline/static_kmin8_5k', 'static_kmin8_5k (static, K_MIN 8)', 'static', '#eda100')]
RANKINGS = [('eval_fisher_k8', 'Ranked by per-token Fisher K=8'), ('eval_static', 'Ranked by cached Fisher (static)'),
            ('eval_write', 'Ranked by write size')]
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'


def curve(path):
    rows = json.loads(Path(path).read_text())['curve']
    rows = [r for r in rows if 0 < r['kept_components'] < 7104]
    return [r['kept_components'] for r in rows], [r['kl'] for r in rows]


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.grid(True, which='major', color=GRID, linewidth=.8); ax.set_axisbelow(True)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main(out):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=False)
    vx, vy = curve(f'{VPD[0]}/klkeep.json')
    for ax, (folder, title) in zip(axes, RANKINGS):
        ax.plot(vx, vy, color=MUTED, linewidth=2, linestyle='--', marker='o', markersize=4, label=VPD[1])
        for run, label, sensor, colour in TRAINED:
            path = Path(run)/folder/'klkeep.json'
            if not path.exists(): continue
            x, y = curve(path)
            own = folder == {'fisher': 'eval_fisher_k8'}.get(sensor, f'eval_{sensor}')
            ax.plot(x, y, color=colour, linewidth=2.6 if own else 1.6, marker='o', markersize=5 if own else 3.5,
                    label=label)
        style(ax, title)
        ax.set_xlabel('kept pieces per token', color=MUTED, fontsize=9)
        ax.set_ylim(.05, 40)
    axes[0].set_ylabel('KL(intact || masked), 128 val rows', color=MUTED, fontsize=9)
    # Adversarial: KL after 320 global attack steps, each run at its own ranking (VPD: CI top-k).
    ax = axes[3]
    keeps = ['top0.005', 'top0.01', 'top0.1']
    series = [(VPD[1], f'{VPD[0]}/adversarial_global_128.json', MUTED, '--')]
    for run, label, sensor, colour in TRAINED:
        series.append((label, f'{run}/adversarial_global_128{"" if sensor == "fisher" else "_" + sensor}.json', colour, '-'))
    for label, path, colour, ls in series:
        if not Path(path).exists(): continue
        kl = json.loads(Path(path).read_text())['kl']
        ax.plot([float(k[3:]) * 100 for k in keeps], [kl[k]['320'] for k in keeps], color=colour, linestyle=ls,
                linewidth=2, marker='o', markersize=5, label=label)
    style(ax, 'Adversarial KL, 320 steps (own ranking)')
    ax.set_xlabel('kept share of pieces (%)', color=MUTED, fontsize=9)
    ax.set_ylim(.05, 40)
    seen = {}
    for a in axes[:3]:
        for h, l in zip(*a.get_legend_handles_labels()): seen.setdefault(l, h)
    fig.legend([h for h in seen.values()], list(seen), loc='lower center', ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle('SimpleStories KLKeep vs VPD (lower is better); thick line = the ranking that run was trained with',
                 color=INK, fontsize=12, x=.01, ha='left')
    fig.tight_layout(rect=(0, .1, 1, .95))
    fig.savefig(out, dpi=150, facecolor='white')
    print('SAVED', out)


if __name__ == '__main__':
    main(sys.argv[1])
