"""Calibration-only budget refinement and a thresholded previous-fit reference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_evaluate as experiment  # noqa: E402
import parauto_structured as structured  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_io import load_pieces  # noqa: E402
from parauto_target import (  # noqa: E402  # noqa: E402
    atomic_json,
    fingerprint,
    load_target,
    sample,
)


@torch.no_grad()
def budget_thresholds(scores, meaningful, budgets=(8, 16, 32, 64)):
    """Threshold quantiles of calibration scores target an average count exactly.

    Scores across matrices share output RMS units. Count monotonicity is safe
    to use; no assumption of accuracy monotonicity is made. Ties may make an
    exact target count impossible, so retain and report the measured count.
    """
    if not meaningful.any():
        raise ValueError("Budget calibration requires meaningful events")
    flat = torch.cat([s[meaningful].flatten() for s in scores])
    n = int(meaningful.sum())
    result = {}
    for budget in budgets:
        number = round(n * budget)
        if number >= flat.numel():
            result[budget] = -1.0
        elif number <= 0:
            result[budget] = float(flat.max())
        else:
            result[budget] = float(flat.topk(number + 1, sorted=False).values.min())
    return result


@torch.no_grad()
def refined_evaluation(target, calibration, test, pieces, seed):
    scores_cal = experiment.all_scores(calibration, pieces, target)
    x = calibration["x"]
    meaningful = (calibration["output"] - (x @ target.W_E) @ target.W_U).square().mean(-1) > toy.EPSILON
    budgets = budget_thresholds(scores_cal, meaningful)
    grid = sorted(set([-1.0, 0.0] + [10 ** (-6 + i / 16) for i in range(97)] + list(budgets.values())))
    curve = []
    for tau in grid:
        row = experiment.check_masks(target, calibration, pieces, [s > tau for s in scores_cal])
        row["threshold_rms"] = tau
        curve.append(row)
    selected = experiment.select_operating_points(curve)
    scores_test = experiment.all_scores(test, pieces, target)
    evaluated = {tau: experiment.check_masks(target, test, pieces, [s > tau for s in scores_test],
                                             probes=8, seed=seed)
                 for tau in {r["threshold_rms"] for r in selected.values()}}
    global_mask = core.topk_mask(torch.cat(scores_test, dim=-1), 32)
    global_keeps = list(global_mask.split([p.count for p in pieces], dim=-1))
    return {"calibration_curve": curve, "budget_thresholds": budgets,
            "posthoc_global_topk32": experiment.check_masks(target, test, pieces, global_keeps),
            "operating_points": {name: {"threshold_rms": row["threshold_rms"], "calibration": row,
                                         "test": evaluated[row["threshold_rms"]]}
                                 for name, row in selected.items()}}


import parauto_tracking as tracking

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--calibration-events', type=int, default=4096)
    parser.add_argument('--test-events', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    tracking.start_cli(args)
    root = Path(args.run)
    original = json.loads((root / 'result.json').read_text())
    if fingerprint(args.target) != experiment.TARGET_SHA or original['target_sha256'] != experiment.TARGET_SHA:
        raise ValueError('Target mismatch')
    target = load_target(args.target, args.device)
    pieces = load_pieces(root, target, args.device)
    caches = {}
    for name, n, offset in [('calibration', args.calibration_events, 41001), ('test', args.test_events, 71001)]:
        generator = torch.Generator(device=args.device).manual_seed(args.seed + offset)
        caches[name] = structured.cache_events(target, sample(n, 100, generator, args.device))
    result = refined_evaluation(target, caches['calibration'], caches['test'], pieces, args.seed)
    result.update(method='projected_lp', target_sha256=experiment.TARGET_SHA, config=vars(args))
    atomic_json(result, root / 'refined_result.json')
    print('THRESHOLD_EVALUATION_COMPLETE', flush=True)
    return result


if __name__ == '__main__':
    tracking.run_main(main)
