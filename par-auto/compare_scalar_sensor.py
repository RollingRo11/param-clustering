"""Matched toy scalar-loss experiment, crossing training and ranking sensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch

import parauto_scalar as scalar
import parauto_structured as structured
import parauto_toy as toy
from parauto_evaluate import TARGET_SHA, all_scores
from parauto_io import load_pieces
from parauto_target import atomic_json, atomic_save, fingerprint, load_target
from evaluate_klkeep import removal_ranks, keep_masks, evaluate_curve, summarize


CURVES = (
    ('output_output', 'Output training / output ranking', '#2171b5'),
    ('scalar_scalar', 'Scalar training / scalar ranking', '#e87517'),
    ('scalar_output', 'Scalar training / output ranking', '#26966d'),
    ('output_scalar', 'Output training / scalar ranking', '#9a5bc5'),
)


def draw(result, root):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, metric, label in zip(axes, ('kl_target', 'mse_target'),
                                ('KL after softmax (T=1)', 'Raw-output MSE vs frozen target')):
        for key, name, color in CURVES:
            curve = result['methods'][key]
            ax.plot(curve['kept'][:-1], curve['metrics'][metric]['mean'][:-1],
                    color=color, label=name, linewidth=2, linestyle='-')
        ax.set(xlabel='Components retained across all four matrices', ylabel=label, xlim=(1, 1600))
        ax.set_xscale('log')
        ax.set_xticks([1, 4, 16, 64, 256, 1600])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yscale('symlog', linthresh=1e-10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, loc='lower left')
    fig.suptitle('Scalar task-loss gradient versus exact output-effect sensor\n'
                 '50k steps · 400 components/matrix · frozen 10k-step toy · 4,096 identical held-out inputs\n'
                 'Unconditional projected Adam · random projected initialization')
    for suffix in ('png', 'svg'):
        fig.savefig(root / f'klkeep.{suffix}', dpi=180)
    plt.close(fig)


def report(result, root):
    meta = result['training']['scalar']
    baseline = result['training']['output']
    lines = ['# Scalar task-loss sensor versus output-effect sensor', '', '![KLKeep comparison](klkeep.png)', '',
             'Both decompositions: 50,000 scheduled updates, 400 pieces per matrix, seed 0, '
             'batch 2,048, 16,384 training inputs, cosine LR 1e-3 to 1e-5, p from 1 to 0.5 '
             'over the first 80%, and unconditional projected Adam. '
             'FP32 training/cache/Adam, FP64 projection, TF32 disabled. The frozen target was trained for 10k steps.', '',
             'The scalar sensor is the absolute directional derivative of each example\'s mean squared '
             'task error against x + ReLU(x), using the verified original labels and equal feature weights. '
             'One true backward pass caches all four matrix-output gradients per batch. No random probes. '
             'There is no cancellation cap, feasibility warmup, backtracking or update rollback. '
             'Both runs start from identically seeded random free factors projected to the target matrices. '
             'Only the training sensor differs; normalization uses each sensor on the same fixed 256 training examples.', '',
             'Four solid curves cross training sensor and analysis sensor. All use the same saved 4,096 '
             'held-out inputs, frozen per-example global component rankings, binary masks, and fresh nonlinear '
             'forwards at every one-component removal level. No ranking is recomputed after removal. '
             'The output-sensor control is freshly trained with the same simplified optimizer and initialization.', '',
             '| Retained | Output/output KL | Scalar/scalar KL | Scalar/output KL | Output/scalar KL |',
             '|---:|---:|---:|---:|---:|']
    for kept in (4, 8, 16, 32, 64, 100, 400, 1600):
        values = [result['methods'][key]['selected_budgets'][str(kept)]['kl_target'] for key, _, _ in CURVES]
        lines.append(f'| {kept} | ' + ' | '.join(f'{v:.5g}' for v in values) + ' |')
    lines += ['', '| Retained | Native scalar/output KL ratio | Paired KL difference ± standard error |',
              '|---:|---:|---:|']
    for kept, row in result['native_paired_comparison'].items():
        lines.append(f"| {kept} | {row['scalar_to_output_kl_ratio']:.4g} | {row['mean_difference']:.5g} ± {row['standard_error']:.3g} |")
    lines += ['', f"Scalar training: {meta['fit_seconds']:.2f} seconds; peak fit allocation "
              f"{meta['peak_fit_allocated_bytes']/2**30:.3f} GiB; cache {meta['cache_bytes']/2**20:.2f} MiB. "
              f"Setup including gradient cache and projected initialization: {meta['setup_seconds']:.2f} seconds.", '',
              f"Matched output training: {baseline['fit_seconds']:.2f} seconds; "
              f"peak {baseline['peak_fit_allocated_bytes']/2**30:.3f} GiB; "
              f"cache {baseline['cache_bytes']/2**20:.2f} MiB. Both fits ran sequentially on the same V100 with the same timing boundaries.", '',
              'Scope: one seed. Scalar scores are in task-MSE derivative units, whereas output scores are '
              'in output-RMS units. The existing epsilon=1e-6 and diagnostic threshold=1e-3 are unchanged; '
              'their relative scales therefore differ. The fixed threshold never gates training or KLKeep. '
              'KL uses temperature-1 softmax over regression outputs; raw-output MSE is shown alongside it. '
              'Scalar derivatives can vanish near task solutions even when finite ablations cause damage. '
              'These curves test cumulative ranked removals, not arbitrary subsets.', '',
              'Checkpoint hashes, timing metadata, per-example paired comparisons, full curves, task-MSE '
              'checks at selected budgets and exact-sum diagnostics are retained in result.json.']
    lines += ['', 'Applied optimizer updates:', '']
    for label, run in (('Scalar', meta), ('Output', baseline)):
        lines += [f'- {label}, {name}: {count:,}' for name, count in run['optimizer_updates_per_matrix'].items()]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


import parauto_tracking as tracking

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    ref = json.loads((args.reference / 'result.json').read_text())
    meta = json.loads((args.run / 'result.json').read_text())
    assert fingerprint(args.target) == TARGET_SHA == ref['target_sha256'] == meta['target_sha256']
    assert (meta['steps_per_matrix'], meta['components_per_matrix'], meta['sensor']) == (50000, 400, scalar.SENSOR)
    scalar.verify_label_definition(args.target)
    output_path = args.output_run
    baseline = json.loads((output_path / 'result.json').read_text())
    assert (baseline['steps_per_matrix'], baseline['components_per_matrix'], baseline['config']['fit_events'],
            baseline['batch_per_matrix_update']) == (50000, 400, 16384, 2048)
    assert baseline['sensor'] == 'output_effect'
    assert meta['optimizer_version'] == baseline['optimizer_version'] == 'projected_adam_no_rejection_v1'
    assert meta['training_config'] == baseline['training_config']
    saved = torch.load(args.reference / 'par_auto_events.pt', map_location='cpu', weights_only=True)
    x, meaningful = saved['x'].cuda(), saved['meaningful']
    budgets = (4, 8, 16, 32, 64, 100)
    reference_kl = {}
    del saved
    target = load_target(args.target, 'cuda')
    parts = {'scalar': load_pieces(args.run, target, 'cuda'), 'output': load_pieces(output_path, target, 'cuda')}
    caches = {'scalar': scalar.cache_events(target, x), 'output': structured.cache_events(target, x)}
    result = {k: ref[k] for k in ('events', 'meaningful_events', 'seed', 'target_sha256', 'target_steps',
                                'temperature', 'tie_break', 'kl_definition', 'metric_caveat')}
    result.update(format='parauto_scalar_sensor_v1', training={'scalar': meta, 'output': baseline},
                  checkpoint_sha256={'scalar': fingerprint(args.run / 'pieces.pt'), 'output': fingerprint(output_path / 'pieces.pt')},
                  methods={}, task_mse={}, native_paired_comparison={},
                  scalar_loss_definition=scalar.LOSS, baseline_task_mse=float(scalar.task_loss(target(x), x).mean()))
    for key, _, _ in CURVES:
        training, sensor = key.split('_')
        pieces = parts[training]
        scores = all_scores(caches[sensor], pieces, target)
        ranks = removal_ranks(scores)
        task = {}
        for kept in (*budgets, 400, 1600):
            masks = [m[0].float() for m in keep_masks(ranks, torch.tensor([1600-kept], device='cuda'), [400]*4)]
            output = toy.forward_with_pieces(target, x, pieces, masks)[0]
            task[str(kept)] = float(scalar.task_loss(output, x).mean())
        result['task_mse'][key] = task
        print(f'START {key}', flush=True)
        values, full = evaluate_curve(target, x, pieces, ranks, batch=128, level_batch=8, temperature=1.)
        torch.testing.assert_close(full, target(x), atol=2e-5, rtol=1e-5)
        result['methods'][key] = summarize(values, meaningful, [400]*4, scores)
        if key == 'output_output':
            reference_kl = {k: values['kl_target'][:, 1600-k].clone() for k in budgets}
        if key == 'scalar_scalar':
            for kept in budgets:
                scalar_kl = values['kl_target'][:, 1600-kept]
                difference = scalar_kl - reference_kl[kept]
                result['native_paired_comparison'][str(kept)] = dict(
                    scalar_to_output_kl_ratio=float(scalar_kl.mean()/reference_kl[kept].mean().clamp_min(1e-30)),
                    mean_difference=float(difference.mean()), standard_error=float(difference.std()/len(x)**.5))
        atomic_save(dict(x=x.cpu(), meaningful=meaningful, removal_ranks=ranks.cpu(),
                         scores=[s.cpu() for s in scores], **values), args.output / f'{key}_events.pt')
        atomic_json(result, args.output / 'result.json')
    atomic_json(result, args.output / 'result.json')
    draw(result, args.output)
    report(result, args.output)
    print('SCALAR_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    tracking.run_main(main)
