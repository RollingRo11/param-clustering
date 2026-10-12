"""Figure for adversarial_probe.py: KL vs adversarial steps, per-position and shared-mask attacks."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
runs = [('vpd_published', 'VPD published', '#c0392b'), ('vpd_paired', 'VPD paired', '#e67e22'), ('parauto_1b', 'par-auto', '#2471a3')]
style = {'native': ('-', 'o'), 'top0.01': ('--', 's'), 'top0.1': (':', '^')}
fig, ax = plt.subplots(1, 3, figsize=(18, 4.3), sharey=True)
for a, scope, title in zip(ax, ('global_128', 'shared', 'position'), ('VPD eval setup: one mask per component, 128 sequences', 'One mask per 8 sequences', 'Separate mask at every token')):
    for key, label, color in runs:
        path = src/f'{key}_{scope}.json'
        if not path.exists(): continue
        kl = json.loads(path.read_text())['kl']
        for gate, (ls, mk) in style.items():
            if gate not in kl: continue
            steps = sorted((int(s) for s in kl[gate] if s.isdigit()))
            name = {'native': 'own CI gate', 'top0.01': 'top 1% kept', 'top0.1': 'top 10% kept'}[gate]
            a.plot([max(s, 1) for s in steps], [kl[gate][str(s)] for s in steps], ls=ls, marker=mk, color=color, label=f'{label}, {name}')
            print(scope, key, gate, {s: round(kl[gate][str(s)], 3) for s in steps}, 'hard zero', round(kl[gate]['hard_zero'], 3))
    if scope == 'global_128':
        paper = {20: .828, 40: 1.354, 80: 3.838, 160: 25.26, 320: 40.22}
        a.plot(list(paper), list(paper.values()), 'k*', ms=10, label='VPD paper table (published)')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('adversary steps (0 plotted at 1)'); a.set_title(title); a.grid(alpha=.3)
ax[0].set_ylabel('KL to real model'); ax[0].legend(fontsize=7)
fig.tight_layout(); dst.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dst, dpi=150)
