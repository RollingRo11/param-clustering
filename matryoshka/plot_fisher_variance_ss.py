"""Bars per matrix from fisher_variance_ss.py: across-token variance of the log read factor |r.h|, of the log write (Fisher)
factor sqrt(u^T F(x) u), and of the write factor's piece-specific part (each token's level shared by the matrix's
pieces removed), sampling noise subtracted. Usage: plot_fisher_variance_ss.py JSON OUT.png"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_ss_klkeep import INK, MUTED, GRID

d = json.load(open(sys.argv[1])); names = list(d)
labels = [f'L{n[2]} {n.split(".")[-1].replace("_proj", "")}' for n in names]
x = np.arange(len(names)); w = .27
fig, ax = plt.subplots(figsize=(12, 4.6))
for off, key, colour, label in ((-w, 'read', '#2a78d6', 'read factor |r.h|'), (0, 'write', '#eb6834', 'write (Fisher) factor, total'),
                                (w, 'write_piece', '#1baf7a', 'write factor, piece-specific (token-shared level removed)')):
    vals = [d[n][key] for n in names]
    ax.bar(x + off, vals, w, color=colour, edgecolor='white', linewidth=1.5, label=f'{label}  (mean {np.mean(vals):.2f})')
ax.axhline(np.mean([d[n]['noise'] for n in names]), color=MUTED, linestyle=':', linewidth=1.2, label='sampling noise (K=8, mean)')
ax.set_xticks(x, labels, fontsize=9, color=INK); ax.set_ylabel('variance across tokens (log factor)', color=MUTED, fontsize=9)
ax.grid(True, axis='y', color=GRID); ax.set_axisbelow(True)
for side in ('top', 'right'): ax.spines[side].set_visible(False)
for side in ('left', 'bottom'): ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=9); ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
ax.set_title('Recipe B: what varies across tokens in the per-token Fisher score (each token\'s top 10% of pieces)', color=INK, fontsize=11, loc='left')
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=150, facecolor='white'); print('SAVED', sys.argv[2])
