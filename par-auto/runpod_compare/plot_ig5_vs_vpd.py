"""Plot completed native-sensor evaluations; training budgets are unmatched."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/ig5_vs_vpd_2000'

def main():
    ig = json.loads((ROOT/'out/ig5_retraining/comparison.json').read_text())['curves']['ig5_train_ig_eval']
    vpd = json.loads((ROOT/'out/runpod_paired_1b/klkeep_2000/comparison.json').read_text())['vpd']
    for key in ('data_sha256', 'target_sha256', 'positions'):
        assert ig[key] == vpd[key], key
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for curve, label, color in [
        (vpd, 'VPD · 2,000 steps · 65.5M training positions', '#d97706'),
        (ig, 'Par-auto IG5 · 900 steps · 29.5M training positions', '#2563eb'),
    ]:
        rows = [r for r in curve['curve'] if r['kept_components'] > 0]
        for ax, key in zip(axes, ('kept_components', 'keep_fraction')):
            ax.plot([r[key] for r in rows], [r['kl'] for r in rows],
                    label=label, color=color, marker='o', markersize=4, linewidth=2)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_ylabel('KL(target || masked), nats / position ↓')
        ax.grid(alpha=.2, which='both')
        ax.legend(fontsize=8, loc='best')
    axes[0].set_xlabel('Components kept per position (all 24 matrices)')
    axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle('KLKeep: VPD at 2,000 steps vs par-auto IG5 at 900 steps', fontsize=14)
    note = ('Same 65,536 held-out positions; native CI / IG5 ranking. Training budgets differ.\n'
            'VPD: 38,912 components, A100 evaluation. IG5: 43,008 components, H100 evaluation; integration error ≈61%.')
    fig.supxlabel(note, fontsize=8)
    for ax in axes:
        ax.set_ylim(bottom=0)
    fig.savefig(OUT/'klkeep_ig5_900_vs_vpd_2000.png', dpi=180)
    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylim(3e-4, 100)
    fig.savefig(OUT/'klkeep_ig5_900_vs_vpd_2000_logy.png', dpi=180)
    plt.close(fig)
    (OUT/'comparison.json').write_text(json.dumps({'ig5':ig, 'vpd':vpd}, indent=2)+'\n')

if __name__ == '__main__':
    main()
