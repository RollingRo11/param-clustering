"""Recipe B against VPD on the dense evals (dense_evals.sh): KLKeep at 21 budgets; faithfulness vs robustness at 10 kept
shares (x = KL with dropped pieces off, y = KL after 320 global attack steps); KL against attack step (every 20) at
0.5 / 1 / 10% kept. Usage: plot_ss_b_vs_vpd_dense.py B_DIR VPD_DENSE_DIR OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_ss_klkeep import INK, MUTED, curve, style

b, v, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
series = [('VPD s-9e774a82 (CI)', v/'klkeep.json', v/'adversarial_global_128.json', '#8a8984', '--'),
          ('Recipe B', b/'dense_klkeep/klkeep.json', b/'dense_adversarial.json', '#2a78d6', '-')]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
for label, kk, _, colour, ls in series:
    x, y = curve(kk)
    a1.plot(x, y, color=colour, linestyle=ls, linewidth=2.2, marker='o', markersize=3.5, label=label)
style(a1, 'KLKeep, 21 budgets (lower is better)'); a1.set_ylim(.1, 40)
a1.set_xlabel('kept pieces per token', color=MUTED, fontsize=9); a1.set_ylabel('KL(intact || masked)', color=MUTED, fontsize=9)
shade = {'top0.005': 1., 'top0.01': .7, 'top0.1': .4}
for label, _, probe, colour, ls in series:
    kl = json.loads(Path(probe).read_text())['kl']
    keys = sorted((k for k in kl if k.startswith('top')), key=lambda k: float(k[3:]))
    xs, ys = [kl[k]['hard_zero'] for k in keys], [kl[k]['320'] for k in keys]
    a2.plot(xs, ys, color=colour, linestyle=ls, linewidth=1.5, marker='o', markersize=6, label=label)
    for k, x, y in zip(keys, xs, ys):
        if k in ('top0.002', 'top0.005', 'top0.01', 'top0.03', 'top0.1'):
            a2.annotate(f'{float(k[3:]) * 100:g}%', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=8, color=INK)
    for k, alpha in shade.items():
        steps = sorted((int(s) for s in kl[k] if s.isdigit()))
        a3.plot(steps, [kl[k][str(s)] for s in steps], color=colour, linestyle=ls, linewidth=1.8, alpha=alpha,
                label=f'{label.split(" ")[0] if label.startswith("VPD") else "B"}, {float(k[3:]) * 100:g}% kept')
style(a2, 'Faithfulness vs robustness, 10 kept shares (lower-left is better)')
a2.set_xlabel('KL with dropped pieces off (no attack)', color=MUTED, fontsize=9); a2.set_ylabel('KL after 320 attack steps', color=MUTED, fontsize=9)
style(a3, 'KL against attack steps (global probe)'); a3.set_xscale('linear')
a3.set_xlabel('attack steps', color=MUTED, fontsize=9); a3.set_ylabel('KL', color=MUTED, fontsize=9)
for ax in (a1, a2): ax.legend(frameon=False, fontsize=9, labelcolor=INK)
a3.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2)
fig.tight_layout(); fig.savefig(out, dpi=150, facecolor='white'); print('SAVED', out)
