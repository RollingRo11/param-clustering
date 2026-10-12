"""Train the parameter autoencoder on the verified frozen 10k-step toy."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import parauto_core as core
import parauto_method as method  # noqa: E402
import parauto_structured as structured  # noqa: E402
import parauto_scalar as scalar
import parauto_tracking as tracking
import parauto_toy as toy  # noqa: E402
from parauto_evaluate import TARGET_SHA  # noqa: E402
from parauto_target import (  # noqa: E402  # noqa: E402
    atomic_json,
    atomic_save,
    fingerprint,
    load_target,
    sample,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--components", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--fit-events", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--sensor", choices=("output_effect", "scalar_task_mse"), default="output_effect")
    args = parser.parse_args(argv)
    tracking.start_cli(args)
    if fingerprint(args.target) != TARGET_SHA:
        raise ValueError("Expected the verified 10,000-step target")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.precision == "fp16" and not args.device.startswith("cuda"):
        raise ValueError("FP16 training requires a CUDA GPU")
    torch.manual_seed(args.seed)
    target = load_target(args.target, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    engine = method
    config = dict(steps=args.steps, batch=args.batch, seed=args.seed, lr=args.lr,
                  lr_min=args.lr_min, precision=args.precision,
                  log_interval=max(1, args.steps // 10))
    cfg = engine.TrainConfig(**config)
    tracking.configure({'training_config': asdict(cfg)})
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.monotonic()
    gen = torch.Generator(device=args.device).manual_seed(args.seed + 101)
    label_provenance = scalar.verify_label_definition(args.target) if args.sensor == scalar.SENSOR else None
    cache_builder = scalar.cache_events if args.sensor == scalar.SENSOR else structured.cache_events
    cache = cache_builder(target, sample(args.fit_events, 100, gen, args.device))
    named = toy.matrices(target)
    tracking.configure(matrix_names=[name for name, _ in named])
    entries = [cache["matrices"][name] for name, _ in named]
    weights = [weight for _, weight in named]
    initial, initialization = [], {}
    for name, weight in named:
        item, info = engine.prepare_initial(weight, args.components, cfg)
        initial.append(item)
        initialization[name] = info
        tracking.log_initialization(name, info)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    fit_started = time.monotonic()
    pieces, histories = engine.fit(entries, weights, initial, cfg,
        log=lambda row: print(json.dumps({"matrix": named[row["matrix_index"]][0], **row}), flush=True))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    fit_seconds = time.monotonic() - fit_started
    peak_bytes = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
    overflow_skips = histories[0][-1].get("overflow_skipped_updates", 0)
    atomic_save({"pieces": [{"read": item.read.cpu(), "write": item.write.cpu()} for item in pieces]}, output / "pieces.pt")
    atomic_json({name: rows for (name, _), rows in zip(named, histories, strict=True)}, output / "training.json")
    result = {"method": "lp_decomposition", "optimizer": "projected_adam", "target_sha256": TARGET_SHA, "target_training_steps": 10000,
              "sensor": args.sensor, "scalar_loss_definition": scalar.LOSS if args.sensor == scalar.SENSOR else None,
              "optimizer_version": "projected_adam_no_rejection_v1",
              "initialization_method": "random_free_factors_then_exact_sum_projection",
              "optimizer_updates_per_matrix": {name: args.steps-overflow_skips for name, _ in named},
              "label_provenance_sha256": label_provenance,
              "optimizer_source_sha256": fingerprint(engine.__file__),
              "precision": {"score_matmuls": args.precision, "norms_loss_normalization": "fp32",
                            "parameters_adam_cache": "fp32", "projection": "fp64", "initialization": "fp32"},
              "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "schedule": {"first_lr": core.cosine_lr(0, cfg), "last_lr": core.cosine_lr(cfg.steps-1, cfg),
                           "p_final": cfg.p_end, "p_anneal_steps": cfg.steps*.8},
              "config": vars(args), "training_config": asdict(cfg), "initialization": initialization,
              "components_per_matrix": args.components, "batch_per_matrix_update": args.batch,
              "fit_seconds": fit_seconds, "setup_seconds": fit_started-started,
              "peak_fit_allocated_bytes": peak_bytes,
              "steps_per_matrix": args.steps, "matrix_updates": args.steps * len(weights),
              "optimizer_calls": args.steps - overflow_skips, "overflow_skipped_updates": overflow_skips,
              "fit_layout": "joint", "cache_backend": "structured",
              "cache_bytes": structured.cache_bytes(cache),
              "elapsed_seconds": time.monotonic() - started}
    atomic_json(result, output / "result.json")
    return result


if __name__ == "__main__":
    tracking.run_main(main)
