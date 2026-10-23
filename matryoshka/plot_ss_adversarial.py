"""Bar chart of the global adversarial probe (KL after 320 attack steps, 128 val sequences) for the SimpleStories runs
and VPD, one panel per kept share, each run at its own ranking (VPD: CI top-k). Usage: plot_ss_adversarial.py OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_ss_klkeep import INK, MUTED, GRID, RUNS, TRAINED, VPD

KEEPS = [('top0.005', '0.5% kept'), ('top0.01', '1% kept'), ('top0.1', '10% kept')]


def main(out):
    series = [(VPD[1], f'{VPD[0]}/adversarial_global_128.json', '#8a8984')]
    for run, label, sensor, colour in TRAINED:
        series.append((label, f'{run}/adversarial_global_128{"" if sensor == "fisher" else "_" + sensor}.json', colour))
    series = [(l, json.loads(Path(p).read_text())['kl'], c) for l, p, c in series if Path(p).exists()]
    fig, axes = plt.subplots(1, len(KEEPS), figsize=(13, 4.4))
    for ax, (key, title) in zip(axes, KEEPS):
        values = [kl[key]['320'] for _, kl, _ in series]
        bars = ax.bar(range(len(series)), values, width=.72, color=[c for *_, c in series], edgecolor='white', linewidth=2)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9, color=INK)
        ax.set_title(title, color=INK, fontsize=11, loc='left')
        ax.set_xticks([]); ax.set_ylim(0, max(values) * 1.15)
        ax.grid(True, axis='y', color=GRID, linewidth=.8); ax.set_axisbelow(True)
        for side in ('top', 'right'): ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'): ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel('KL(intact || attacked mask)', color=MUTED, fontsize=9)
    fig.legend(axes[0].patches, [l for l, *_ in series], loc='lower center', ncol=len(series), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle('Global adversarial probe, 320 steps, 128 val sequences, each run at its own ranking (lower is better)',
                 color=INK, fontsize=12, x=.01, ha='left')
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(out, dpi=150, facecolor='white')
    print('SAVED', out)


if __name__ == '__main__':
    main(sys.argv[1])
