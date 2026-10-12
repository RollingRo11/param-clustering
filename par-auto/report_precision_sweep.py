"""Report completed precision ablations and failures without treating failure as speed."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    tracking.start_cli(args)
    labels = {'baseline': 'Current mixed FP16', 'parameters_fp16': 'FP16 params / grads / Adam',
              'activations_fp16': 'FP16 activations', 'projection_fp32': 'FP32 projection + backward'}
    results = {key: json.loads((args.root/key/'result.json').read_text()) for key in labels if (args.root/key/'result.json').exists()}
    (args.root/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    lines = ['# 5k-step precision speed sweep', '',
             'Each ablation changes one precision group relative to the current mixed-FP16 score-matmul baseline. Runs execute sequentially in fresh Python processes on the same V100 allocation. All use seed 0, the verified 10k-step target, 400 components per matrix, batch 2,048, and 5k cosine LR / p schedules. No KLKeep sweep is run.', '',
             '| Variant | Status | Fit seconds | Warm ms/step | Peak allocated GiB |', '|---|---|---:|---:|---:|']
    for key, r in results.items():
        if r['status'] == 'completed':
            lines.append(f"| {labels[key]} | Complete | {r['fit_seconds']:.2f} | {r['warm_ms_per_step']:.3f} | {r['peak_allocated_bytes']/2**30:.3f} |")
        else:
            lines.append(f"| {labels[key]} | Failed: {r['error']} | — | — | — |")
    lines += ['', 'Fit timing excludes target/cache construction, initialization and checkpoint saving. Warm timing excludes the first 200 optimizer-step boundaries; it includes the final cap/projection. Peak allocated memory includes live target, cache, factors, gradients, optimizer state and temporary tensors. No custom kernels or compilation.', '',
              'All variants preserve stock Adam epsilon 1e-8 and the existing matrix-sum tolerance of 1e-5. FP16 parameter storage cannot generally retain that exactness after rounding, and FP16 Adam moments also risk underflow. A failed run is not given a throughput estimate.', '']
    for key, r in results.items():
        lines += [f'## {labels[key]}', '', f"Initial maximum matrix-sum relative error: {max(r['initial_sum_errors'].values()):.4g}.",
                  f"Scheduled step boundaries completed: {r['scheduled_steps']}; Adam calls: {r['adam_updates']}.", '']
        if r['status'] == 'completed':
            lines += [f"Final maximum sum error: {max(r['final_sum_errors'].values()):.4g}. Overflow-skipped updates: {r['overflow_skipped_updates']}.", '']
        if key == 'activations_fp16': lines += [f"Activation scope: {r['activation_scope']}. Fixed reference scores stay FP32. With cache_and_intermediates, large saved score intermediates and vector norms are FP16; only reduced norms are promoted to FP32.", '']
        if key == 'parameters_fp16': lines += [r['scaler_note'], '']
        if r['status'] == 'failed': lines += [r['error'], '']
    (args.root/'report.md').write_text('\n'.join(lines))
    completed = [(key, r) for key, r in results.items() if r['status'] == 'completed']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, field, factor, title in ((axes[0], 'warm_ms_per_step', 1, 'Warm step time (ms)'), (axes[1], 'peak_allocated_bytes', 1/2**30, 'Peak allocated tensors (GiB)')):
        values = [r[field]*factor for _, r in completed]
        bars = ax.barh([labels[k] for k, _ in completed], values, color=['#64748b', '#1676b6', '#dc781c'][:len(completed)])
        ax.bar_label(bars, fmt='%.3f', padding=4); ax.invert_yaxis(); ax.set_xlim(0, max(values)*1.2); ax.set_title(title); ax.grid(axis='x', alpha=.2)
    fig.suptitle('Precision speed ablations · 5k steps each · same V100\nFailed configurations excluded from performance bars')
    for ext in ('png', 'svg'): fig.savefig(args.root/f'profile.{ext}', dpi=180)
    plt.close(fig)
    print('PRECISION_SWEEP_REPORT_COMPLETE', flush=True)


if __name__ == '__main__': tracking.run_main(main)
