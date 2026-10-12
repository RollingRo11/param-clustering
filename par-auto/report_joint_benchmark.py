"""Check the scheduling comparison and summarize its measured tradeoff."""
import argparse
import json
from pathlib import Path

import torch


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    runs = {name: json.loads((args.root / name / "result.json").read_text())
            for name in ("sequential", "joint", "spd")}
    for key in ("steps_per_matrix", "components_per_matrix", "batch_per_matrix_update", "target_sha256", "gpu"):
        assert len({run[key] for run in runs.values()}) == 1
    steps = runs["joint"]["steps_per_matrix"]
    assert runs["joint"]["optimizer_calls"] == steps
    assert runs["sequential"]["optimizer_calls"] == 4 * steps
    assert all(run["matrix_updates"] == 4 * steps for run in runs.values())
    states = [torch.load(args.root / name / "pieces.pt", map_location="cpu", weights_only=True)["pieces"]
              for name in ("sequential", "joint")]
    equality, max_error = [], 0.
    for a, b in zip(*states, strict=True):
        for key in ("read", "write"):
            equality.append(torch.equal(a[key], b[key]))
            max_error = max(max_error, float((a[key]-b[key]).abs().max()))
    summary = {"runs": runs, "all_eight_factor_tensors_bitwise_identical": all(equality),
               "maximum_factor_absolute_difference": max_error,
               "training_histories_identical": (args.root / "sequential/training.json").read_text() ==
                                                (args.root / "joint/training.json").read_text()}
    (args.root / "comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# One-loss training comparison", "",
             f"{steps:,} updates per matrix; 400 components; batch 2,048; the same allocated V100. "
             "Frozen target: verified 10,000-step residual toy. All three runs use eager PyTorch.", "",
             "| Method | Fit seconds | Setup + fit seconds | Peak fit tensors (GiB) | Optimizer calls |",
             "|---|---:|---:|---:|---:|"]
    for name, run in runs.items():
        fit = run["phases"]["fit"]
        lines.append(f"| {name} | {fit['wall_seconds']:.3f} | {run['fit_plus_setup_seconds']:.3f} | "
                     f"{fit['peak_torch_allocated_bytes']/2**30:.3f} | {run['optimizer_calls']:,} |")
    lines += ["", f"All eight trained factor tensors bitwise identical between method layouts: **{all(equality)}**. "
              f"Maximum absolute difference: {max_error:.3g}. "
              f"Training histories identical: **{summary['training_histories_identical']}**.", "",
              "One loss preserves each matrix's normalization, gradient, batch stream, Adam state, projection "
              "and cancellation cap. It changes scheduling and holds four graphs in memory at once. "
              "A small timing difference in a single short run should not be treated as a reliable speedup. "
              "Cache setup is separate from fit; imports, target loading and post-fit evaluation are excluded.", "",
              "SPD uses the pinned unmodified authors' optimizer. Both method layouts use the same structured cache. "
              "method resamples 16,384 cached training examples while SPD generates fresh examples from the same distribution. "
              "This pilot checks scheduling and resources, not converged decomposition quality.", ""]
    (args.root / "report.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
