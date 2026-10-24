"""Recipe B against VPD: KLKeep curves, and robustness vs faithfulness at the probe's budgets (x = KL with the dropped
pieces off, no attack; y = KL after 320 global attack steps; one point per kept share). Usage: plot_ss_b_vs_vpd.py B_DIR OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_ss_klkeep import INK, MUTED, GRID, VPD, curve, style

b_dir, out = Path(sys.argv[1]), sys.argv[2]
series = [('VPD s-9e774a82 (CI)', Path(VPD[0]), Path(VPD[0])/'klkeep.json', Path(VPD[0])/'adversarial_global_128.json', '#8a8984', '--'),
          ('Recipe B', b_dir, b_dir/'eval_static/klkeep.json', b_dir/'adversarial_global_128_static.json', '#2a78d6', '-')]
fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
for label, _, klkeep, _, colour, ls in series:
    x, y = curve(klkeep)
    left.plot(x, y, color=colour, linestyle=ls, linewidth=2.2, marker='o', markersize=4, label=label)
style(left, 'KLKeep (128 val rows; lower is better)')
left.set_xlabel('kept pieces per token', color=MUTED, fontsize=9); left.set_ylabel('KL(intact || masked)', color=MUTED, fontsize=9)
left.set_ylim(.1, 40); left.legend(frameon=False, fontsize=9, labelcolor=INK)
keeps = [('top0.005', '0.5%'), ('top0.01', '1%'), ('top0.1', '10%')]
for label, _, _, probe, colour, ls in series:
    kl = json.loads(probe.read_text())['kl']
    xs, ys = [kl[k]['hard_zero'] for k, _ in keeps], [kl[k]['320'] for k, _ in keeps]
    right.plot(xs, ys, color=colour, linestyle=ls, linewidth=1.5, marker='o', markersize=7, label=label)
    for (_, name), x, y in zip(keeps, xs, ys):
        right.annotate(name, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8.5, color=INK)
style(right, 'Faithfulness vs robustness at the probe\'s budgets (lower-left is better)')
right.set_xlabel('KL with dropped pieces off (no attack)', color=MUTED, fontsize=9)
right.set_ylabel('KL after 320 global attack steps', color=MUTED, fontsize=9)
right.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout(); fig.savefig(out, dpi=150, facecolor='white'); print('SAVED', out)
