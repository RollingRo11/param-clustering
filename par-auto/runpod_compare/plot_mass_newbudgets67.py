"""Per-piece weight mass, sorted, for slides: the Matryoshka lambda = 1 runs (old and new budgets, 5k steps) against
published VPD (400k steps). mass_c = ||u_c||^2 ||r_c||^2 as a share of the 24 matrices' total ||W||_F^2 (see
weight_mass67.py); orthogonal pieces would sum to 100%. Usage: plot_mass_newbudgets67.py OUTPUT_PNG MASS_JSON ..."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

# Entities keep their colors from the earlier figures (lambda = 1 violet, new budgets blue, VPD green dashed);
# blue/violet/green validated all-pairs.
SERIES = [('matryoshka_lp1_newbudgets_5000', 'ours, λ = 1, new budgets (5k)', '#2a78d6', '-'),
          ('matryoshka_lp1_5000', 'ours, λ = 1 (5k)', '#4a3aa7', '-'),
          ('vpd_published_400000', 'VPD, published (400k)', '#008300', '--')]
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    output = Path(sys.argv[1])
    methods = {}
    for path in sys.argv[2:]: methods.update(json.loads(Path(path).read_text())['methods'])
    fig, ax = plt.subplots(figsize=(8.2, 4.9), dpi=200, facecolor=SURFACE)
    for label, name, color, style in SERIES:
        m = methods[label]
        shares = 100*np.array(m['sorted_shares'])
        ax.plot(np.arange(1, len(shares)+1), shares, color=color, ls=style, lw=2.4,
                label=f"{name}: all pieces add to {100*m['total_share']:.0f}% of the weights")
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(1e-4, 1.2)          # % units: the largest piece is 0.76%; below 0.0001% are only the last tails
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}%')); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}')); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('pieces, largest first', color=INK, fontsize=11)
    ax.set_ylabel('share of the model\'s weight mass', color=INK, fontsize=11)
    ax.set_title('Weight mass per piece', color=INK, fontsize=13, loc='left')
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc='lower left')
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=10)
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for label in [s[0] for s in SERIES] + ['vpd_5000']:
        m = methods.get(label)
        if m: print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
                            if k in ('pieces', 'total_share', 'largest_share', 'pieces_for_50pct', 'pieces_for_90pct', 'residual_share')})


if __name__ == '__main__': main()
