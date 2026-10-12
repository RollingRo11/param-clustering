"""Summary and figure for repair_probe.py.

Added damage A(mode) = KL(U damaged, D in mode) - KL(D in mode alone); A(real) and A(frozen) need no subtraction.
Surviving share = A(mode) / A(real): 1 means masking D passes upstream damage on like the real model; 0 means erased.
Self-repair of the real model = 1 - A(real)/A(frozen).
Uncertainty: bootstrap over sequences (the same resampled sequences for every number).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

src, dst, keep = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else '0.01'
DAMAGES = ['vpd', 'vpd_random', 'parauto', 'noise_matrix', 'noise_resid']
DLABEL = {'vpd': 'VPD mask', 'vpd_random': 'random VPD mask', 'parauto': 'par-auto mask', 'noise_matrix': 'noise on matrix outputs', 'noise_resid': 'noise on residual'}
COLS = ['frozen', 'vpd', 'parauto']        # vpd_random left out: its mask alone already gives KL ~8, so added damage saturates
CLABEL = {'frozen': 'frozen\n(never see damage)', 'vpd': 'VPD mask', 'vpd_random': 'random VPD\nmask', 'parauto': 'par-auto\nmask'}
runs = [(p.stem.replace('vpd_', 'VPD '), json.loads(p.read_text())) for p in sorted(src.glob('vpd_*.json'))]
rng = np.random.default_rng(0)


def shares(d, idx):
    """Per pair: {mode: A(mode)/A(real)} and model self-repair. idx: [B, n] sequence resamples -> arrays [pairs, B]."""
    K = d['kl'][keep]; mean = lambda v: np.asarray(v)[idx].mean(-1)
    out = {dm: {c: [] for c in COLS+['self_repair']} for dm in DAMAGES}
    for p in d['pairs']:
        U, D = '+'.join(map(str, p['U'])), '+'.join(map(str, p['D']))
        for dm in DAMAGES:
            real = mean(K['real'][dm][U])
            for c in COLS:
                a = mean(K['pair'][dm][U][D][c]) - (0 if c == 'frozen' else mean(K['alone'][c][D]))
                out[dm][c].append(a/real)
            out[dm]['self_repair'].append(1-real/mean(K['pair'][dm][U][D]['frozen']))
    return {dm: {c: np.array(v) for c, v in m.items()} for dm, m in out.items()}


fig, axes = plt.subplots(1, len(runs)+1, figsize=(6.2*(len(runs)+1), 4.6))
for ax, (label, d) in zip(axes, runs):
    n = len(d['kl'][keep]['real']['vpd']['0'])
    full = {dm: {c: v[:, 0] for c, v in m.items()} for dm, m in shares(d, np.arange(n)[None]).items()}
    boot = shares(d, rng.integers(0, n, (1000, n)))                              # [pairs, 1000]
    grid = np.array([[np.median(full[dm][c]) for c in COLS] for dm in DAMAGES])
    lo = np.array([[np.percentile(np.median(boot[dm][c], 0), 2.5) for c in COLS] for dm in DAMAGES])
    hi = np.array([[np.percentile(np.median(boot[dm][c], 0), 97.5) for c in COLS] for dm in DAMAGES])
    im = ax.imshow(grid, cmap='RdBu', vmin=-.2, vmax=2.2, aspect='auto')
    for i in range(len(DAMAGES)):
        for j in range(len(COLS)):
            ax.text(j, i, f'{grid[i, j]:.2f}\n[{lo[i, j]:.2f}, {hi[i, j]:.2f}]', ha='center', va='center', fontsize=7.5)
    ax.set_xticks(range(len(COLS)), [CLABEL[c] for c in COLS], fontsize=8); ax.set_yticks(range(len(DAMAGES)), [DLABEL[dm] for dm in DAMAGES], fontsize=8)
    ax.set_title(f'{label}: upstream damage that reaches the output\n(relative to real later layers = 1; median of 17 layer splits)', fontsize=9)
    ax.set_xlabel('later layers are...'); ax.set_ylabel('earlier layers damaged by...')
    print(f'== {label}, keep {keep}, sequences {n}, noise scale {d["noise_scale"][keep]}, check {d["equivalence_check"]}')
    for dm in DAMAGES:
        print(f'  {DLABEL[dm]:24s}', ' '.join(f'{c}={np.median(full[dm][c]):.2f} [{lo[DAMAGES.index(dm), COLS.index(c)]:.2f},{hi[DAMAGES.index(dm), COLS.index(c)]:.2f}]' for c in COLS),
              f'| model self-repair={np.median(full[dm]["self_repair"]):.2f}',
              f'| VPD-mask share <0.5 in {sum(v < .5 for v in full[dm]["vpd"])}/17 splits, par-auto in {sum(v < .5 for v in full[dm]["parauto"])}/17')
# last panel: every split and damage kind, A(real) vs A(masked later layers), all runs
ax = axes[-1]
for (label, d), marker in zip(runs, 'o^'):
    K = d['kl'][keep]
    for c, color in (('vpd', '#c0392b'), ('parauto', '#2471a3')):
        xs, ys = [], []
        for p in d['pairs']:
            U, D = '+'.join(map(str, p['U'])), '+'.join(map(str, p['D']))
            for dm in DAMAGES:
                xs.append(np.mean(K['real'][dm][U])); ys.append(max(np.mean(K['pair'][dm][U][D][c])-np.mean(K['alone'][c][D]), 1e-3))
        ax.scatter(xs, ys, s=16, marker=marker, color=color, alpha=.6, label=f'{label}; later layers: {"VPD" if c == "vpd" else "par-auto"} mask')
ax.plot([1e-2, 30], [1e-2, 30], 'k--', lw=1, label='passes damage on like the real model')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('damage reaching output, real later layers (KL)')
ax.set_ylabel('added damage, masked later layers (KL)'); ax.set_title(f'All 17 splits x 5 damage kinds, {float(keep)*100:g}% kept', fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=.3)
fig.tight_layout(); dst.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dst, dpi=150)
