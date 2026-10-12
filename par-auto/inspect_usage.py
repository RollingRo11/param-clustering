"""Count globally used pieces in the saved decomposition, without retraining."""
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

import parauto_evaluate as experiment  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_io import evaluation_payload, load_pieces  # noqa: E402
from parauto_target import (  # noqa: E402  # noqa: E402
    atomic_json,
    fingerprint,
    load_target,
    sample,
)


import parauto_tracking as tracking

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    root = Path(args.run)
    original = json.loads((root / "result.json").read_text())
    refined = json.loads((root / "refined_result.json").read_text())
    assert fingerprint(args.target) == original["target_sha256"] == experiment.TARGET_SHA
    tau = evaluation_payload(refined)["operating_points"]["pass_0.95"]["threshold_rms"]
    torch.backends.cuda.matmul.allow_tf32 = False
    target = load_target(args.target, "cuda")
    pieces = load_pieces(root, target, "cuda")
    caches, scores = {}, {}
    cfg = original.get("config", {"seed": 0, "calibration_events": 4096, "test_events": 4096})
    for split, offset in (("calibration", 41001), ("test", 71001)):
        gen = torch.Generator(device="cuda").manual_seed(cfg["seed"] + offset)
        x = sample(cfg.get(f"{split}_events", 4096), target.config["n_features"], gen, "cuda")
        caches[split] = toy.cache_events(target, x)
        scores[split] = experiment.all_scores(caches[split], pieces, target)
    thresholds = sorted({1e-8, 1e-6, 1e-4, 1e-3, tau, 1e-2})
    result = {"target_sha256": experiment.TARGET_SHA, "method": "projected_lp", "threshold_rms": tau,
              "events": {k: len(c["x"]) for k, c in caches.items()}, "matrices": {},
              "definition": "A piece is ever retained if its first-order output RMS score exceeds the fixed calibrated threshold on at least one event."}
    for i, ((name, _), p) in enumerate(zip(toy.matrices(target), pieces)):
        cal, test = scores["calibration"][i], scores["test"][i]
        combined = torch.cat([cal, test])
        count = (combined > tau).sum(0)
        profiles = torch.load(Path(args.profiles) / f"profiles_{name}.pt", weights_only=True)
        exact = profiles["exact_ablation_rms"]
        row = {"available": p.count,
               "nonzero_rank_one_matrices": int(((p.read.norm(dim=0) * p.write.norm(dim=0)) > 0).sum()),
               "ever_retained_calibration": int((cal > tau).any(0).sum()),
               "ever_retained_test": int((test > tau).any(0).sum()),
               "ever_retained_combined": int((count > 0).sum()),
               "retained_on_at_least_10_events": int((count >= 10).sum()),
               "retained_on_at_least_1pct_events": int((count >= 0.01 * len(combined)).sum()),
               "ever_above_score_threshold": {str(t): int((combined > t).any(0).sum()) for t in thresholds},
               "isolated_feature_exact_effect_above_threshold": {str(t): int((exact > t).any(0).sum()) for t in thresholds}}
        result["matrices"][name] = row
        print(json.dumps({"matrix": name, **row}), flush=True)
    # Define globally retained sets using calibration only, then remove all
    # other pieces together on held-out test events, without reprojection.
    n = len(caches["test"]["x"])
    union = [(s > tau).any(0).unsqueeze(0).expand(n, -1) for s in scores["calibration"]]
    result["global_calibration_union_pruning"] = experiment.check_masks(target, caches["test"], pieces, union)
    result["global_calibration_union_counts"] = [int(k[0].sum()) for k in union]
    result["fixed_top100_by_calibration_mean_squared_score"] = {}
    chosen = []
    for s in scores["calibration"]:
        mask = torch.zeros_like(s[0], dtype=torch.bool)
        mask[s.square().mean(0).topk(100).indices] = True
        chosen.append(mask.unsqueeze(0).expand(n, -1))
    result["fixed_top100_by_calibration_mean_squared_score"]["all_matrices"] = experiment.check_masks(target, caches["test"], pieces, chosen)
    for i, (name, _) in enumerate(toy.matrices(target)):
        masks = [torch.ones(n, p.count, device="cuda", dtype=torch.bool) for p in pieces]
        masks[i] = chosen[i]
        result["fixed_top100_by_calibration_mean_squared_score"][name] = experiment.check_masks(target, caches["test"], pieces, masks)
    atomic_json(result, Path(args.output) / "result.json")
    print(json.dumps({k: v for k, v in result.items() if k != "matrices"}), flush=True)
    print("USAGE_INSPECTION_COMPLETE", flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
