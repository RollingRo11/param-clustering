"""Plot both completed KLKeep curves together and record matched timing."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, POSITIONS, save_json


def main():
    results, curves = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, color in [('parauto', '#176b46'), ('vpd', '#9250a1')]:
        directory = ROOT/'output/train'/name
        results[name] = json.loads((directory/'result.json').read_text())
        curves[name] = json.loads((directory/'klkeep.json').read_text())
        assert results[name]['positions'] == POSITIONS and results[name]['complete']
        rows = [r for r in curves[name]['curve'] if r['kept_components'] > 0]
        for ax, x in zip(axes, ['kept_components', 'keep_fraction']):
            ax.plot([r[x] for r in rows], [r['kl'] for r in rows], marker='o', color=color, label=name)
            ax.set_xscale('log'); ax.set_ylabel('KL(target || masked), nats / position')
            ax.grid(alpha=.2); ax.legend()
    assert results['vpd']['data_sha256'] == results['parauto']['data_sha256']
    assert curves['vpd']['data_sha256'] == curves['parauto']['data_sha256']
    axes[0].set_xlabel('Components kept per position, across all 24 matrices')
    axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle('1B training positions per method · native-sensor KLKeep')
    fig.savefig(ROOT/'output/klkeep_comparison.png', dpi=180)
    plt.close(fig)
    save_json(dict(training=results, evaluation=curves), ROOT/'output/comparison.json')


if __name__ == '__main__': main()
