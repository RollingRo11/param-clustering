"""Fresh-forward evaluation and training-only threshold calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_method as objectives  # noqa: E402
import parauto_toy as toy  # noqa: E402

TARGET_SHA = "027bc51f51f24ceb6fd75565451fc6e278fab9146949d1279eac61047b949fab"
THRESHOLDS = [-1.0, 0.0] + [10 ** (-6 + index / 4) for index in range(25)]


def log(message):
    print(f"[parauto_evaluate] {message}", flush=True)


@torch.no_grad()
def all_scores(cache, pieces, target):
    return [torch.cat([objectives.scores_of(objectives.slice_entry(cache["matrices"][name], slice(start, start+128)), pieces[i])
                       for start in range(0, len(cache["x"]), 128)])
            for i, (name, _) in enumerate(toy.matrices(target))]


@torch.no_grad()
def check_masks(target, cache, pieces, keeps, probes=0, seed=0):
    x, full = cache["x"], cache["output"]
    meaningful = (full - (x @ target.W_E) @ target.W_U).square().mean(-1) > toy.EPSILON
    masked, _, _ = toy.forward_with_pieces(target, x, pieces, keeps)
    mse = (masked - full).square().mean(-1)
    worst = mse.clone()
    generator = torch.Generator(device=x.device).manual_seed(seed + 456)
    for probe in range(probes):
        restored = []
        for keep in keeps:
            values = torch.rand(keep.shape, generator=generator, device=x.device)
            if probe % 2 == 0:
                values = (values > 0.5).float()
            restored.append(keep.float() + (1 - keep.float()) * values)
        output, _, _ = toy.forward_with_pieces(target, x, pieces, restored)
        worst = torch.maximum(worst, (output - full).square().mean(-1))
    counts = torch.stack([keep.sum(-1) for keep in keeps], -1).float()
    total = counts.sum(-1)
    return {"events": len(x), "meaningful_events": int(meaningful.sum()),
            "pass_meaningful": toy.masked_stat((mse <= toy.EPSILON).float(), meaningful, "mean"),
            "pass_all": float((mse <= toy.EPSILON).float().mean()),
            "mse_mean_meaningful": toy.masked_stat(mse, meaningful, "mean"),
            "mse_p95_meaningful": float(torch.quantile(mse[meaningful], 0.95)) if meaningful.any() else None,
            "kept_mean_all": float(total.mean()),
            "kept_mean_meaningful": toy.masked_stat(total, meaningful, "mean"),
            "kept_p95_meaningful": float(torch.quantile(total[meaningful], 0.95)) if meaningful.any() else None,
            "kept_per_matrix_meaningful": [toy.masked_stat(counts[:, i], meaningful, "mean") for i in range(counts.shape[1])],
            "superset_pass_meaningful": toy.masked_stat((worst <= toy.EPSILON).float(), meaningful, "mean") if probes else None,
            "superset_probes": probes, "max_output_error": float((masked - full).abs().max())}


def select_operating_points(curve, pass_targets=(0.9, 0.95, 0.99), budgets=(8, 16, 32, 64)):
    """Select on calibration only; do not assume threshold accuracy is monotone."""
    if any(row["pass_meaningful"] is None for row in curve):
        raise ValueError("Calibration needs meaningful events")
    selections = {}
    for target in pass_targets:
        candidates = [row for row in curve if row["pass_meaningful"] >= target]
        if candidates:
            selections[f"pass_{target:g}"] = min(candidates, key=lambda r: r["kept_mean_meaningful"])
    for budget in budgets:
        selections[f"budget_{budget}"] = min(curve, key=lambda r: abs(r["kept_mean_meaningful"] - budget))
    return selections


@torch.no_grad()
def evaluate_thresholds(target, calibration, test, pieces, seed):
    scores_cal = all_scores(calibration, pieces, target)
    curve = []
    for tau in THRESHOLDS:
        keeps = [scores > tau for scores in scores_cal]
        row = check_masks(target, calibration, pieces, keeps)
        row["threshold_rms"] = tau
        curve.append(row)
    selected = select_operating_points(curve)
    scores_test = all_scores(test, pieces, target)
    tests = {}
    for tau in sorted({row["threshold_rms"] for row in selected.values()}):
        tests[tau] = check_masks(target, test, pieces, [s > tau for s in scores_test], probes=8, seed=seed)
    # An explicit all-kept control must preserve score-zero pieces too.
    all_kept = check_masks(target, test, pieces, [torch.ones_like(s, dtype=torch.bool) for s in scores_test])
    if all_kept["max_output_error"] > 2e-5:
        raise RuntimeError(f"All-kept forward drift: {all_kept}")
    comparisons = {str(k): check_masks(target, test, pieces, [core.topk_mask(s, k) for s in scores_test])
                   for k in (1, 2, 4, 8, 16, 32)}
    return {"calibration_curve": curve, "all_kept": all_kept, "posthoc_topk": comparisons,
            "operating_points": {name: {"threshold_rms": row["threshold_rms"], "calibration": row,
                                         "test": tests[row["threshold_rms"]]}
                                 for name, row in selected.items()}}


@torch.no_grad()
def cancellation_report(cache, pieces, target, batch=128):
    report = {}
    for (name, weight), p in zip(toy.matrices(target), pieces):
        entry = cache["matrices"][name]
        mass, net = 0.0, 0.0
        for start in range(0, len(entry["h"]), batch):
            part = objectives.slice_entry(entry, slice(start, start + batch))
            mass += float(objectives.scores_of(part, p).sum())
            net += float(objectives.net_effect_rms(part, weight).sum())
        report[name] = {"mean_sum_magnitudes_rms": mass / len(entry["h"]),
                        "mean_net_rms": net / len(entry["h"]), "cancellation_ratio": mass / max(net, 1e-12)}
    return report


