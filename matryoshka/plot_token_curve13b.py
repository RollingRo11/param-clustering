"""KLKeep against training tokens, with the Matryoshka 13B run (KISSKI, VPD's token stream, mass penalty 10) at its kept
snapshots: our steps 1,000 (the earliest kept) / 2,500 / 5,000 / 7,500 / 12,500 / 25,000 / 40,000 / 50,000 / 100,000 = the tokens VPD sees at
4k / 10k / 20k / 30k / 50k / 100k / 160k / 200k / 400k of its steps (one of our steps = 4 of VPD's: 256 vs 64 sequences of 512).
Also the earlier 5k-step Matryoshka run (paired data, new budgets, no mass penalty) at its checkpoints, the paired VPD
run's checkpoints, and the published VPD run (13.1B tokens). Dashed: a straight-line fit in log-log space (KL = a * tokens^b)
to the 13B run's points from FIT_FROM tokens on, extended past the data. Same 65,536 held-out positions, each method
ranked by its own score; one panel per keep fraction.
Usage: plot_token_curve13b.py DATA_DIR OUTPUT_PNG   (DATA_DIR holds the klkeep.json files under the names in main())"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np

M13B_STEPS = (1000, 2500, 5000, 7500, 12500, 25000, 40000, 50000, 100000)
OURS_5K_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)
VPD_PAIRED_STEPS = (1000, 2000, 5000, 10000, 15000, 18664)
FRACTIONS = (0.001, 0.01, 0.1)
FIT_FROM = 1e9                 # fit the 13B run's points at >= 1B tokens (past the early fast drop)
FIT_TO = 1e11                  # draw the fitted line out to here
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#008300'    # validated palette slots: blue = no mass penalty, orange = mass 10,
                                                         # as in plot_mass_adversarial67.py; blue was the 5k run in plot_token_curve67.py
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
DATA_SHA = 'e564bc5744fe056fd468c6ecd84cbf05d8b4d9dcb37ae5f5924faf5510dc9d8a'


def load(path, steps):
    result = json.loads(Path(path).read_text())
    assert result['positions'] == 65536 and result['data_sha256'] == DATA_SHA and result['training_steps'] == steps, path
    return result


def at(result, fraction):
    """(tokens trained, kept pieces, KL) at one keep fraction."""
    row, = [r for r in result['curve'] if abs(r['keep_fraction'] - fraction) < 1e-9]
    return result['training_positions'], int(row['kept_components']), row['kl']


def tokens_label(x, _=None):
    return f'{x/1e9:g}B' if x >= 1e9 else f'{x/1e6:g}M'


def main():
    data, output = Path(sys.argv[1]), Path(sys.argv[2])
    m13b = [load(data/f'm13b_{s:06d}.json', s) for s in M13B_STEPS]
    ours5k = [load(data/f'ours5k_{s}.json', s) for s in OURS_5K_STEPS]
    paired = [load(data/f'vpd_paired_{s}.json', s) for s in VPD_PAIRED_STEPS]
    published = load(data/'vpd_published.json', 400_000)
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(12.4, 4.3), dpi=170, facecolor=SURFACE)
    fits = {}
    for ax, fraction in zip(axes, FRACTIONS):
        for results, color in ((ours5k, BLUE), (m13b, ORANGE)):
            points = [at(r, fraction) for r in results]
            ax.plot([p[0] for p in points], [p[2] for p in points], color=color, lw=2.2, marker='o', ms=3.2, markeredgewidth=0)
        points = np.array([at(r, fraction) for r in m13b], dtype=float)
        use = points[:, 0] >= FIT_FROM
        slope, intercept = np.polyfit(np.log(points[use, 0]), np.log(points[use, 2]), 1)
        xs = np.geomspace(points[use, 0].min(), FIT_TO, 50)
        ax.plot(xs, np.exp(intercept) * xs ** slope, color=ORANGE, lw=1.6, ls=(0, (2, 2)))
        fits[fraction] = dict(exponent=slope, points=int(use.sum()), kl_at_13_1b=float(np.exp(intercept) * 13.1072e9 ** slope),
                              kl_at_100b=float(np.exp(intercept) * 1e11 ** slope))
        points = [at(r, fraction) for r in paired]
        ax.plot([p[0] for p in points], [p[2] for p in points], color=GREEN, lw=2, ls='--', marker='o', ms=3.6, markeredgewidth=0)
        x, vpd_k, kl = at(published, fraction)
        ax.plot([x], [kl], ls='none', marker='o', ms=7, markerfacecolor=SURFACE, markeredgecolor=GREEN, markeredgewidth=2)
        ours_k = at(m13b[-1], fraction)[1]
        ax.set_title(f'{fraction*100:g}% kept  ({ours_k:,} pieces; VPD {vpd_k:,})', color=INK, fontsize=10, loc='left')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1.2e7, 1.4e11)
        ax.xaxis.set_major_locator(FixedLocator([1e7, 1e8, 1e9, 1e10, 1e11]))
        ax.xaxis.set_major_formatter(tokens_label); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1., 2., 3., 5.)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}')); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('tokens trained', color=INK, fontsize=9)
        ax.set_facecolor(SURFACE)
        ax.grid(True, which='major', color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ('top', 'right'): ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'): ax.spines[side].set_color(GRID)
    axes[0].set_ylabel('KL to the real model', color=INK, fontsize=9)
    handles = [
        Line2D([], [], color=ORANGE, lw=2.2, marker='o', ms=3.2, markeredgewidth=0, label="ours, 13B run: new budgets + mass penalty 10, VPD's tokens"),
        Line2D([], [], color=ORANGE, lw=1.6, ls=(0, (2, 2)), label='fit to the 13B run from 1B tokens, extended'),
        Line2D([], [], color=BLUE, lw=2.2, marker='o', ms=3.2, markeredgewidth=0, label='ours, 5k run: new budgets, no mass penalty, paired data'),
        Line2D([], [], color=GREEN, lw=2, ls='--', marker='o', ms=3.6, markeredgewidth=0, label='VPD, paired run (checkpoints)'),
        Line2D([], [], ls='none', marker='o', ms=7, markerfacecolor=SURFACE, markeredgecolor=GREEN, markeredgewidth=2,
               label='VPD, published run (13.1B tokens)')]
    fig.suptitle('KLKeep vs tokens trained (65,536 held-out tokens)', color=INK, fontsize=11, x=0.01, y=0.99, ha='left')
    fig.legend(handles=handles, loc='upper left', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.0, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', facecolor=SURFACE)
    print('wrote', output, flush=True)
    for fraction in FRACTIONS:
        print(f'{fraction*100:g}% kept', '13B run', [(f'{at(r, fraction)[0]/1e9:.2f}B', round(at(r, fraction)[2], 3)) for r in m13b],
              'VPD published', round(at(published, fraction)[2], 3), 'fit', {k: round(v, 4) for k, v in fits[fraction].items()})


if __name__ == '__main__': main()
