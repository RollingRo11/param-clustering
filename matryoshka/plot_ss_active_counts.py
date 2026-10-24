"""Grouped bars: mean active pieces per token in each matrix, floorset30's above-floor set (split_check_ss.py output)
against VPD s-9e774a82's CI > 0 set. Usage: plot_ss_active_counts.py SPLIT_JSON OUT.png"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_ss_klkeep import INK, MUTED, GRID

VPD = [1.0, 1.2, 0.8, 1.5, 5.4, 5.1, 1.9, 2.0, 2.7, 6.3, 9.2, 10.1]   # s-9e774a82, CI > 0, 16k val tokens
split = json.load(open(sys.argv[1]))
names = list(split['above_floor_per_matrix'])
ours = [split['above_floor_per_matrix'][n] for n in names]
labels = [f'L{n[2]} {n.split(".")[-1].replace("_proj", "")}' for n in names]
x = np.arange(len(names)); w = .38
fig, ax = plt.subplots(figsize=(11, 4.2))
for off, vals, colour, label in ((-w / 2, ours, '#eda100', f'floorset30, above floor (total {sum(ours):.0f})'),
                                 (w / 2, VPD, '#8a8984', f'VPD, CI > 0 (total {sum(VPD):.0f})')):
    bars = ax.bar(x + off, vals, w, color=colour, edgecolor='white', linewidth=2, label=label)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + .15, f'{v:.1f}', ha='center', fontsize=8, color=INK)
ax.set_xticks(x, labels, fontsize=9, color=INK)
ax.set_ylabel('active pieces per token (mean)', color=MUTED, fontsize=9)
ax.grid(True, axis='y', color=GRID); ax.set_axisbelow(True)
for side in ('top', 'right'): ax.spines[side].set_visible(False)
for side in ('left', 'bottom'): ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=9)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)
ax.set_title('Active pieces per token by matrix: floorset30 (step 5,000) vs VPD', color=INK, fontsize=12, loc='left')
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=150, facecolor='white'); print('SAVED', sys.argv[2])
