"""Build the matched-run report from completed benchmark artifacts (CPU SLURM)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from parauto_io import method_directory


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--released", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Optional original dense method result.json")
    args = parser.parse_args()
    tracking.start_cli(args)
    args.output.mkdir(parents=True, exist_ok=True)
    method_root = method_directory(args.full)
    runs = {name: json.loads((path / "result.json").read_text()) for name, path in (("spd", args.full / "spd"), ("method", method_root))}
    released = json.loads((args.released / "result.json").read_text())
    for run in runs.values():
        assert run["steps_per_matrix"] == 50000
        assert run["components_per_matrix"] == 400
        assert run["batch_per_matrix_update"] == 2048
        assert run["matrix_updates"] == 200000
    for key in ("gpu", "target_sha256", "torch", "cuda", "precision"):
        assert runs["spd"][key] == runs["method"][key]
    sf, bf = (runs[name]["phases"]["fit"] for name in ("spd", "method"))
    summary = {"spd": runs["spd"], "method": runs["method"], "released": released,
               "speedup_method_over_SPD": sf["wall_seconds"] / bf["wall_seconds"],
               "allocated_memory_ratio_method_over_SPD": bf["peak_torch_allocated_bytes"] / sf["peak_torch_allocated_bytes"]}
    if args.baseline:
        summary["dense_method_baseline"] = json.loads(args.baseline.read_text())
    training = json.loads((method_root / "training.json").read_text())
    summary["method_update_audit"] = {name: {key: rows[-1][key] for key in ("step", "optimizer_updates", "rejected_updates", "backtracks") if key in rows[-1]}
                                 for name, rows in training.items()}
    assert all(rows[-1]["step"] == 49999 for rows in training.values())
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharex=True, sharey=True)
    variants = [("Released SPD", args.released, released),
                ("SPD retrain", args.full / "spd", runs["spd"]),
                ("Projected Lp", method_root, runs["method"])]
    profiles = [torch.load(path / "figure10_profiles.pt", map_location="cpu", weights_only=True)
                for _, path, _ in variants]
    high = max(float(torch.stack([p["target"].max(), p["matched"].max()]).max()) for p in profiles) * 1.08
    low = min(-.03, min(float(torch.stack([p["target"].min(), p["matched"].min()]).min()) for p in profiles) * 1.1)
    for ax, (name, _, run), profile in zip(axes, variants, profiles, strict=True):
        for layer, color in enumerate(("#4464de", "#f07167")):
            ax.scatter(profile["target"][layer].numpy().ravel(), profile["matched"][layer].numpy().ravel(),
                       s=9, alpha=.4, color=color, edgecolors="none", label=f"Layer {layer+1}", rasterized=True)
        metric = run["evaluation"]["figure10"]
        ax.plot([low, high], [low, high], "--", color=".65", linewidth=1.5)
        ax.text(.04, .95, f"r = {metric['pearson_r']:.5f}\nRelative RMSE = {metric['relative_rmse']:.3f}\nSlope = {metric['slope_through_origin']:.3f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox={"facecolor": "white", "edgecolor": ".85", "alpha": .9})
        ax.set_title(name)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
        ax.set_xlabel("Target neuron contribution")
    axes[0].set_ylabel("Selected component neuron contribution")
    axes[-1].legend(loc="lower right")
    fig.suptitle("50,000 steps • 400 components per matrix • frozen 10,000-step target", fontsize=14)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"figure10_comparison.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    values = [("Training time (minutes)", [sf["wall_seconds"]/60, bf["wall_seconds"]/60]),
              ("Peak tensor memory (GiB)", [sf["peak_torch_allocated_bytes"]/2**30, bf["peak_torch_allocated_bytes"]/2**30]),
              ("Mean GPU utilization (%)", [sf["mean_gpu_utilization_pct"], bf["mean_gpu_utilization_pct"]])]
    for ax, (title, numbers) in zip(axes, values, strict=True):
        bars = ax.bar(["SPD", "Our method"], numbers, color=["#4464de", "#e79a3f"], width=.55)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in numbers], padding=5)
        ax.set_ylim(0, max(numbers)*1.2)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Matched 50k runs on the same V100", fontsize=14)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"resource_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = ["# SPD and projected Lp: matched 50,000-step comparison", "",
             "Both runs completed on the same allocated Tesla V100-SXM2-32GB, sequentially in separate processes. "
             "Each used 400 rank-one components per matrix, batch 2,048 and 50,000 updates per matrix across four matrices. "
             "The frozen target is the verified 10,000-step, 100-feature, two-layer residual MLP.", "",
             "| Measurement | SPD retrain | Our method |", "|---|---:|---:|"]
    for title, sv, bv in [
        ("Timed fit (seconds)", sf["wall_seconds"], bf["wall_seconds"]),
        ("Fit + cache/initialization (seconds)", runs["spd"]["fit_plus_setup_seconds"], runs["method"]["fit_plus_setup_seconds"]),
        ("Peak allocated tensors during fit (GiB)", sf["peak_torch_allocated_bytes"]/2**30, bf["peak_torch_allocated_bytes"]/2**30),
        ("Peak allocated tensors including setup (GiB)",
         max(p["peak_torch_allocated_bytes"] for p in runs["spd"]["phases"].values())/2**30,
         max(p["peak_torch_allocated_bytes"] for p in runs["method"]["phases"].values())/2**30),
        ("Peak PyTorch reserved (GiB)", sf["peak_torch_reserved_bytes"]/2**30, bf["peak_torch_reserved_bytes"]/2**30),
        ("Peak NVIDIA process/device memory (MiB)", sf["peak_nvidia_memory_used_mib"], bf["peak_nvidia_memory_used_mib"]),
        ("Mean GPU utilization (%)", sf["mean_gpu_utilization_pct"], bf["mean_gpu_utilization_pct"]),
        ("Mean memory controller utilization (%)", sf["mean_memory_controller_utilization_pct"], bf["mean_memory_controller_utilization_pct"]),
        ("Mean power (watts)", sf["mean_power_watts"], bf["mean_power_watts"]),
    ]:
        lines.append(f"| {title} | {sv:.3f} | {bv:.3f} |")
    if runs["method"].get("cache_backend") == "structured":
        lines.extend(["", "method uses exact structured scoring in ordinary eager PyTorch: one shared final-output "
                      "Jacobian, cached 25-dimensional input QR factors, and a shared-plus-masked "
                      "low-dimensional first-output calculation. No custom kernel or compiler is used. "
                      "The recorded run configuration identifies the objective and optimizer version.", "",
                      f"Resident structured cache: {runs['method']['cache_bytes']/2**20:.3f} MiB (distinct tensor storages)."])
    if args.baseline:
        old = summary["dense_method_baseline"]["phases"]["fit"]
        lines.extend(["", f"Previous dense method baseline: {old['wall_seconds']/60:.3f} minutes and "
                      f"{old['peak_torch_allocated_bytes']/2**30:.3f} GiB peak allocated tensors during fit. "
                      f"The new fit is {old['wall_seconds']/bf['wall_seconds']:.3f} times faster. "
                      "That earlier run used a separate allocation; the SPD/new-method comparison above shares one allocation."])
    lines.extend(["", "![Matched resource comparison](resource_comparison.png)", "",
                  "| Figure 10 metric | Released SPD | SPD retrain | Our method |", "|---|---:|---:|---:|"])
    for key, title in [("pearson_r", "Pearson r"), ("relative_rmse", "Relative RMSE"), ("slope_through_origin", "Slope through origin")]:
        vals = [r["evaluation"]["figure10"][key] for r in (released, runs["spd"], runs["method"])]
        lines.append(f"| {title} | " + " | ".join(f"{v:.6f}" for v in vals) + " |")
    counts = [" / ".join(str(r["evaluation"]["per_layer"][str(i)]["distinct_selected_pieces"]) for i in range(2))
              for r in (released, runs["spd"], runs["method"])]
    lines.append("| Distinct selected input pieces, layers 1 / 2 | " + " | ".join(counts) + " |")
    lines.extend(["", "![Figure 10 comparison](figure10_comparison.png)", "",
        "## What was matched and what differs", "",
        "SPD uses the authors' unmodified `spd-paper` optimizer, losses, gates, initialization and released configuration, "
        "pinned at commit `c6314c9f702b81af593927025aa0ae5aaed4ca4c`. W&B uploads and image generation are disabled. "
        "Native print diagnostics and final checkpoint saving remain inside the timed region. The fresh run is not expected "
        "to be bitwise identical to the released run: disabling plots changes random-number consumption, and both benchmarks "
        "use the shared PyTorch 2.7.1/CUDA 12.6 runtime. The released checkpoint separately verifies the paper replication.", "",
        ("One optimizer call updates all four matrices in both methods. method sums its separately normalized matrix losses "
         "and uses one backward call and Adam step, followed by each matrix's independent projection. "
         if runs["method"].get("fit_layout") == "joint" else
         "One SPD optimizer call updates all four matrices. method fits the four matrices sequentially: 50,000 calls for each, "
         "200,000 calls total. ") +
        "Both therefore perform 200,000 matrix updates. Timings cover the complete four-matrix fit; "
        "a 'step' is not silently counted four times differently. Each matrix sees 102.4 million batch rows during fitting.", "",
        "method samples batches from a fixed 16,384-example training-only activation/Jacobian cache; SPD generates fresh inputs "
        "during training. They use the same input distribution, not identical examples or equal unique-data counts. "
        "method's cache and initialization are measured separately and included in the setup-plus-fit total. "
        "The method fit includes the final factor/history save. Both exclude imports, target loading and post-training evaluation.", "",
        "Trainable factors and activations use float32 with TF32, autocast and compilation disabled. method retains its existing "
        "float64 projection solves/residual arithmetic. Memory includes all resident tensors, including method's cache. "
        "GPU utilization/power are sampled every 200 ms with NVIDIA telemetry; wall time is CUDA-synchronized. "
        "Utilization is an activity percentage, not a FLOP count. This is one timed seed per method, not an uncertainty estimate.", "",
        "method retains the previously tested Lp objective (p anneals from 1 to 0.5), cosine learning rate, exact-sum projection, "
        "with exact-sum projection. Its count and batch changed from the earlier 10k experiment (1,024 and 128) to match SPD "
        "(400 and 2,048). Thus this is not a controlled step-count-only comparison with that earlier run. "
        "400 exceeds the matrices' rank/MLP width of 25, though it is below the residual width of 1,000; the free-factor "
        "projection supports it without a change to the algorithm.", "",
        "## Interpreting Figure 10", "",
        "The calculation follows equations 11–12 and the authors' plotting code: choose one input component per feature "
        "per layer by its largest signed neuron contribution, then plot that component's entire 25-neuron profile. "
        "There are 5,000 plotted feature–neuron pairs. It is not a pointwise envelope. The harness checks both contribution "
        "tensors against the authors' routines for the released and newly trained SPD checkpoints.", "",
        "Correlation alone does not measure scale accuracy. Relative RMSE is ||selected-target|| / ||target||; "
        "the slope reveals systematic amplification or attenuation. This analysis evaluates component/neuron structure, "
        "not sparse-forward sufficiency. method's all-kept reconstruction is exact by construction; SPD uses a soft faithfulness loss, "
        "so that all-kept check is not a like-for-like sparse selection benchmark.", "",
        "Sources: [SPD paper](https://arxiv.org/html/2506.20790v1), "
        "[pinned authors' optimizer](https://github.com/goodfire-ai/param-decomp/blob/c6314c9f702b81af593927025aa0ae5aaed4ca4c/spd/run_spd.py), "
        "[released decomposition](https://wandb.ai/goodfire/spd-resid-mlp/runs/wau744ht).", "",
        "Full numeric evaluations, executed configurations, timing phases and source/target identifiers are in `comparison.json` "
        "and the individual run directories. GPU telemetry CSVs retain the raw samples.", ""])
    (args.output / "report.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(args.output), "speedup_method_over_SPD": summary["speedup_method_over_SPD"]}), flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
