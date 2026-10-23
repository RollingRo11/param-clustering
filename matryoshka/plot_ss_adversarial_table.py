"""The global adversarial probe as tables in one image: KL after 0 (switched off), 20, 40, 80, 160 and 320 attack steps,
one table per kept share, a row per run at its own ranking (VPD: CI). Usage: plot_ss_adversarial_table.py OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_ss_klkeep import INK, MUTED, GRID, TRAINED, VPD
from plot_ss_adversarial import KEEPS

STEPS = [('hard_zero', 'no attack'), ('20', '20'), ('40', '40'), ('80', '80'), ('160', '160'), ('320', '320')]


def main(out):
    series = [(VPD[1], f'{VPD[0]}/adversarial_global_128.json', '#8a8984')]
    for run, label, sensor, colour in TRAINED:
        series.append((label, f'{run}/adversarial_global_128{"" if sensor == "fisher" else "_" + sensor}.json', colour))
    series = [(l, json.loads(Path(p).read_text())['kl'], c) for l, p, c in series if Path(p).exists()]
    fig, axes = plt.subplots(len(KEEPS), 1, figsize=(10, 1.15 + 1.25 * len(KEEPS) * (len(series) + 1) / 5))
    for ax, (key, title) in zip(axes, KEEPS):
        ax.axis('off')
        ax.set_title(f'{title}: KL after n attack steps', color=INK, fontsize=11, loc='left')
        cells = [[f'{kl[key][s]:.2f}' for s, _ in STEPS] for _, kl, _ in series]
        best = [min(range(len(series)), key=lambda i: series[i][1][key][s]) for s, _ in STEPS]
        table = ax.table(cellText=cells, rowLabels=[l for l, *_ in series], colLabels=[h for _, h in STEPS],
                         loc='center', cellLoc='center', rowLoc='left')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.45)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(GRID); cell.get_text().set_color(INK if r else MUTED)
            if c == -1 and r > 0:
                cell.get_text().set_color(series[r - 1][2]); cell.get_text().set_fontweight('bold')
            if r > 0 and c >= 0 and best[c] == r - 1: cell.get_text().set_fontweight('bold')
    fig.suptitle('Global adversarial probe, 128 val sequences, each run at its own ranking (lower is better; bold = best)',
                 color=INK, fontsize=12, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print('SAVED', out)


if __name__ == '__main__':
    main(sys.argv[1])
