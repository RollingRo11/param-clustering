"""Layer-local error vs pieces kept per token: ours against published VPD."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
COUNTS = [43, 129, 193, 512, 1290, 4301, 12903]
KINDS = [('linear', 'linear writes (v, o, down)', 'tab:blue'),
         ('gelu', 'after the GELU (c_fc)', 'tab:red'),
         ('attention', 'attention pattern (q, k)', 'tab:green')]


def series(name, kind):
    xs, ys = [], []
    for c in COUNTS:
        path = SRC/f'{name}_{c}.json'
        if path.exists():
            d = json.loads(path.read_text())
            xs.append(d['mean_kept_per_token']); ys.append(d['by_kind'][kind]['mean_local'])
    return xs, ys


fig, ax = plt.subplots(figsize=(7.2, 4.6))
for kind, label, colour in KINDS:
    ax.plot(*series('gim_5000', kind), color=colour, marker='o', label=f'ours: {label}')
    ax.plot(*series('vpd_published', kind), color=colour, marker='s', linestyle='--', alpha=.7,
            label=f'VPD: {label}')
native = json.loads((SRC/'vpd_published_native.json').read_text())
ax.axvline(native['mean_kept_per_token'], color='grey', lw=1, ls=':')
ax.text(native['mean_kept_per_token']*1.06, 1.22, "VPD's own support\n(217 pieces/token, KL 0.29)",
        fontsize=8, color='grey', va='top')
ax.set_xscale('log')
ax.set_xlabel('pieces kept per token (of 43,008 ours / 38,912 VPD)')
ax.set_ylabel('local error, relative to the matrix output')
ax.set_title("Does the kept set reproduce each matrix's own output?")
ax.set_ylim(0, 1.35)
ax.grid(alpha=.3)
ax.legend(fontsize=7.5, ncol=2, loc='lower left')
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f'wrote {OUT}')
