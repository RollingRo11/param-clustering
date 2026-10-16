"""Plot and upload both matched-step KLKeep curves without changing training runs."""
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from common import ROOT, save_json, credentials, upload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=2000)
    parser.add_argument('--output', type=Path, default=ROOT/'output')
    args = parser.parse_args()
    credentials(); curves = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, color in [('parauto', '#176b46'), ('vpd', '#9250a1')]:
        directory = args.output/name
        curves[name] = json.loads((directory/'klkeep.json').read_text())
        assert curves[name]['training_steps'] == args.step
        rows = [r for r in curves[name]['curve'] if r['kept_components'] > 0]
        for ax, x in zip(axes, ['kept_components', 'keep_fraction']):
            ax.plot([r[x] for r in rows], [r['kl'] for r in rows], marker='o', color=color, label=name)
            ax.set_xscale('log'); ax.set_ylabel('KL(target || masked), nats / position')
            ax.grid(alpha=.2); ax.legend()
    for key in ('training_data_sha256', 'data_sha256', 'target_sha256', 'positions', 'training_positions', 'precision'):
        assert curves['vpd'][key] == curves['parauto'][key], key
    axes[0].set_xlabel('Components kept per position, across all 24 matrices')
    axes[1].set_xlabel('Fraction of components kept')
    fig.suptitle(f'{args.step:,} training updates each · native-sensor KLKeep')
    figure = args.output/f'klkeep_{args.step}.png'
    fig.savefig(figure, dpi=180); plt.close(fig)
    save_json(curves, args.output/'comparison.json')
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu',
                     name=f'paired-1b-klkeep-{args.step}-comparison', group='paired-h200-1b-20260910', job_type='evaluation',
                     config=dict(training_steps=args.step, training_positions=args.step*32768))
    run.log({f'eval/klkeep_{args.step}': wandb.Image(str(figure))})
    save_json(dict(id=run.id, url=run.url), args.output/'wandb.json')
    run.finish()
    receipts = []
    for path in sorted(args.output.rglob('*')):
        if path.is_file() and 'wandb' not in path.parts and path.suffix in ('.json', '.png'):
            receipts.append(upload(path, f'evaluations/{args.step}/'+str(path.relative_to(args.output))))
    save_json(receipts, args.output/'r2_receipts.json')


if __name__ == '__main__': main()
