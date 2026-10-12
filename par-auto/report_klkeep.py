"""Render a completed native-sensor ablation run without rerunning model forwards."""
import argparse
import json
from pathlib import Path
from evaluate_klkeep import draw


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    result = json.loads((args.run/'result.json').read_text())
    draw(result, args.run)
    ours, spd = [result['methods'][key] for key in ('par_auto', 'spd')]
    lines = ['# Native-sensor KLKeep: matched 50k decompositions', '',
             'Both fits have 400 independent rank-one components per matrix (1,600 total). '
             'Par-auto is the combined-loop run 10242577; SPD is the fresh retrain in run 10242461. '
             'The frozen target was trained for 10,000 steps.', '',
             f"Evaluated all 1,601 integer budgets on the same {result['events']:,} fresh inputs "
             f"(seed {result['seed']}; {result['meaningful_events']:,} meaningful under the existing MSE > 1e-4 criterion). "
             'Every removal level uses a fresh nonlinear forward.', '',
             'Par-auto ranks components by its output-effect RMS sensor. SPD uses the authors’ learned causal '
             'importances, including their native lower-leaky clamp. Each example has its own global ranking across '
             'four matrices, frozen on the original target activations. Least-important pieces are removed first; '
             'exact score ties use stable matrix/component index. Masks are binary for both methods. '
             'No unrepresented weight residual or continuous CI scaling is added.', '',
             '**Metric:** KL(softmax(target output) || softmax(masked output)), temperature 1. '
             'This toy predicts continuous values, so softmax is an analysis convention. Raw-output MSE is '
             'reported alongside KL. All means below include all fresh examples, including zero-feature inputs.', '',
             'X-axes are logarithmic. Zero-component endpoints remain in the table and data but are omitted from the log plots.', '',
             '![KLKeep curves](klkeep.png)', '',
             '| Retained total | Par-auto KL | SPD KL | Par-auto MSE | SPD MSE |',
             '|---:|---:|---:|---:|---:|']
    for kept in (0, 4, 8, 16, 32, 64, 100, 400, 1600):
        a, b = [entry['selected_budgets'][str(kept)] for entry in (ours, spd)]
        lines.append(f"| {kept} | {a['kl_target']:.3g} | {b['kl_target']:.3g} | {a['mse_target']:.3g} | {b['mse_target']:.3g} |")
    lines += ['', f"Par-auto reaches and stays below 1% of all-removed KL from retained budget **{ours['normalized_target_thresholds']['kl_target']['0.01']['all_larger_budgets_pass_from']}**, "
              f"and 1% of all-removed MSE from budget **{ours['normalized_target_thresholds']['mse_target']['0.01']['all_larger_budgets_pass_from']}**. "
              f"SPD's first passing budgets are {spd['normalized_target_thresholds']['kl_target']['0.01']['first_kept']} (KL) and "
              f"{spd['normalized_target_thresholds']['mse_target']['0.01']['first_kept']} (MSE); None means no budget passed. "
              'SPD’s nonzero all-kept reconstruction error is included.', '',
              'SPD is non-monotonic: some removals improve target reconstruction. The own-reference curves below '
              'separately measure removal damage relative to each decomposition’s full output, without mixing in '
              'its initial discrepancy from the target. These also show considerable damage under SPD’s ranking '
              'well before reaching the small retained budgets.', '',
              '![Damage relative to own full output](own_reference.png)', '',
              '![Removal order](removal_curve.png)', '',
              'Validation: nested masks remove exactly one component at a time; batched masked forwards match '
              'independently reconstructed weight-matrix ablations on a small fixture; KL direction checked. '
              'On the real checkpoint, the shared SPD forward matches the authors’ code at 0, 800 and 1,600 removals. '
              'All-removed outputs match the residual-only target and all-kept par-auto matches the frozen target. '
              'All three new tests passed on allocation 10242938.', '',
              'This measures cumulative prefixes of one native-sensor ranking per example. It does not establish '
              'that every subset of low-score components can be removed harmlessly, or a globally shared small subset '
              'for all inputs. SPD exact ties can make the curve depend on the documented tie order. '
              'The full results retain per-budget standard errors and meaningful-input means/p95; per-example '
              'metrics and rankings are preserved on Explorer in par_auto_events.pt and spd_events.pt.', '']
    (args.run/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[11:23]), flush=True)
    print('KLKEEP_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    tracking.run_main(main)
