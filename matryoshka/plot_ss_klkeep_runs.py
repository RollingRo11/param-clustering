"""One-panel KLKeep (each run at its own ranking) for a list of SimpleStories runs against VPD.
Usage: plot_ss_klkeep_runs.py OUT.png NAME=RUN_DIR ...  (RUN_DIR holds eval_static/klkeep.json)"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_ss_klkeep import INK, MUTED, GRID, VPD, curve, style

COLOURS = ['#eda100', '#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#008300', '#7a5af0', '#c43d3d']   # reference order
out, specs = sys.argv[1], [a.split('=', 1) for a in sys.argv[2:]]
fig, ax = plt.subplots(figsize=(9, 5.4))
x, y = curve(f'{VPD[0]}/klkeep.json')
ax.plot(x, y, color='#8a8984', linestyle='--', linewidth=2.2, marker='o', markersize=4, label=VPD[1])
for (name, run), colour in zip(specs, COLOURS):
    x, y = curve(Path(run)/'eval_static/klkeep.json')
    ax.plot(x, y, color=colour, linewidth=2, marker='o', markersize=4, label=name)
style(ax, 'KLKeep, each run at its own ranking (128 val rows; lower is better)')
ax.set_xlabel('kept pieces per token', color=MUTED, fontsize=9); ax.set_ylabel('KL(intact || masked)', color=MUTED, fontsize=9)
ax.set_ylim(.1, 40); ax.axvline(71, color=GRID, linewidth=1.5, zorder=0)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc='lower left')
fig.tight_layout(); fig.savefig(out, dpi=150, facecolor='white'); print('SAVED', out)
