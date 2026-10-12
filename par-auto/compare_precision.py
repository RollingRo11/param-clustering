"""Evaluate FP16-trained par-auto in FP32 against the existing FP32 50k run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch

import parauto_structured as structured
from parauto_evaluate import TARGET_SHA, all_scores
from parauto_io import load_pieces
from parauto_target import load_target, fingerprint, atomic_json, atomic_save
from evaluate_klkeep import removal_ranks, evaluate_curve, summarize


def draw(result, root):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, metric, title in zip(axes, ('kl_target', 'mse_target'), ('KL after softmax (T=1)', 'Raw-output MSE')):
        for key, label, color in (('fp32', 'FP32 training · 50k', '#2171b5'), ('fp16', 'FP16 mixed training · 50k', '#e87517')):
            curve = result['methods'][key]
            ax.plot(curve['kept'][:-1], curve['metrics'][metric]['mean'][:-1], label=label, color=color, linewidth=2, linestyle='-')
        ax.set(title=title, ylabel='Mean error', xlabel='Components retained (all four matrices)', xlim=(1, 1600))
        ax.set_xscale('log'); ax.set_xticks([1, 4, 16, 64, 256, 1600]); ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yscale('symlog', linthresh=1e-10); ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.legend(loc='lower left')
    fig.suptitle('Par-auto precision comparison · native-sensor KLKeep · 400 components/matrix\nFrozen 10k-step target · 4,096 shared held-out inputs · FP32 evaluation')
    for ext in ('png', 'svg'): fig.savefig(root / f'klkeep.{ext}', dpi=180)
    plt.close(fig)


import parauto_tracking as tracking

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    reference = json.loads((args.reference / 'result.json').read_text())
    meta = json.loads((args.run / 'result.json').read_text())
    assert fingerprint(args.target) == TARGET_SHA == reference['target_sha256'] == meta['target_sha256']
    assert meta['steps_per_matrix'] == 50000 and meta['components_per_matrix'] == 400
    assert meta['config']['precision'] == 'fp16'
    root = args.run / 'comparison'; root.mkdir(exist_ok=True)
    saved = torch.load(args.reference / 'par_auto_events.pt', map_location='cpu', weights_only=True)
    x, meaningful = saved['x'].cuda(), saved['meaningful']; del saved
    target = load_target(args.target, 'cuda')
    parts = load_pieces(args.run, target, 'cuda')
    assert all(p.read.dtype == p.write.dtype == torch.float32 for p in parts)
    cache = structured.cache_events(target, x)
    scores = all_scores(cache, parts, target); del cache
    ranks = removal_ranks(scores)
    values, full = evaluate_curve(target, x, parts, ranks, batch=128, level_batch=8, temperature=1.)
    torch.testing.assert_close(full, target(x), atol=2e-5, rtol=1e-5)
    baseline = json.loads((Path(reference['checkpoints']['par_auto']).parent / 'result.json').read_text())
    result = {k: reference[k] for k in ('events', 'meaningful_events', 'seed', 'target_sha256', 'target_steps', 'temperature', 'tie_break', 'kl_definition', 'metric_caveat')}
    result.update(format='parauto_precision_v1', evaluation_precision='fp32; FP64 KL/MSE accumulation; TF32 disabled',
                  sensor='output-effect RMS, frozen per-input global ranking before cumulative binary removal',
                  methods={'fp32': reference['methods']['par_auto'], 'fp16': summarize(values, meaningful, [p.count for p in parts], scores)},
                  training={'fp32': baseline, 'fp16': meta},
                  checkpoints={'fp32': reference['checkpoints']['par_auto'], 'fp16': str(args.run / 'pieces.pt')},
                  checkpoint_sha256={'fp32': reference['checkpoint_sha256']['par_auto'], 'fp16': fingerprint(args.run / 'pieces.pt')})
    atomic_save({'x': x.cpu(), 'meaningful': meaningful, 'removal_ranks': ranks.cpu(), 'scores': [s.cpu() for s in scores], **values}, root / 'fp16_events.pt')
    atomic_json(result, root / 'result.json'); draw(result, root)
    lines = ['# Par-auto: FP32 versus FP16 mixed training', '', '![Precision comparison](klkeep.png)', '',
             'Both runs: 50k scheduled steps, seed 0, 400 components per matrix, batch 2,048, cosine LR 1e-3 to 1e-5, p 1 to 0.5 over the first 80%, and the verified 10k-step target.', '',
             'FP16 is used for training score matmuls with dynamic gradient scaling. Parameters, Adam states, cache, loss/norm reductions, initialization and cancellation audit remain FP32. Exact-sum projections remain FP64. Both checkpoints are evaluated in FP32, using the same saved 4,096 inputs and their own output-effect sensors. KL and MSE accumulation use FP64.', '',
             '| Retained | FP32 KL | FP16 KL | FP32 MSE | FP16 MSE |', '|---:|---:|---:|---:|---:|']
    for kept in (4, 8, 16, 32, 64, 100, 400, 1600):
        cells = [result['methods'][key]['selected_budgets'][str(kept)][metric] for metric in ('kl_target', 'mse_target') for key in ('fp32', 'fp16')]
        lines.append(f'| {kept} | ' + ' | '.join(f'{v:.4g}' for v in cells) + ' |')
    lines += ['', f"FP16 fit: {meta['fit_seconds']:.2f} s; peak allocated tensors {meta['peak_fit_allocated_bytes']/2**30:.3f} GiB; GPU {meta['gpu']}; gradient-overflow skipped updates: {meta['overflow_skipped_updates']}.", '',
              'Reference FP32 fit: 663.67 s, peak allocated tensors 0.766 GiB, V100. These are separate V100 allocations and one seed, not a controlled throughput benchmark; historical timing includes final saves while the new standalone fit timer excludes them.', '',
              'The x axis is logarithmic; zero-retained remains in JSON. KL uses temperature-1 softmax over toy regression outputs. This evaluates cumulative sensor-ranked removals, not arbitrary subsets.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    print('PRECISION_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__': tracking.run_main(main)
