"""Scalar CINDER for the Goodfire-scope Pile-67M target.

Every scalar in the 24 attention/MLP matrices is a node and receives exactly
one hard owner.  The discovery feature for scalar j is a matrix-free random
projection of its gate-Hessian row.  For a parameter probe r, it is estimated
by finite gradient transport

    [dF/dm_j (m + rho r) - dF/dm_j (m - rho r)] / (2 rho).

Several event sign sketches and output contrasts are evaluated together with
batched VJPs.  The token bank is copied into host memory once and reused for
all probes; the fitting loop performs no token-file I/O.

The 4096-way result is produced by recursive balanced two-seed cuts in the
curvature-fingerprint space.  Coarser 2048- and 1024-way decompositions are
nested prefixes of the same tree.  There is no background component, soft
ownership, SVD atom, PCA, centroid reservoir, or ablation-trained loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# Parfact implementations live in the sibling parfact directory.
import sys as _parfact_sys
from pathlib import Path as _ParfactPath
for _parfact_dir in (_ParfactPath(__file__).resolve().parent / "parfact",
                     _ParfactPath(__file__).resolve().parent.parent / "parfact"):
    if _parfact_dir.is_dir():
        _parfact_sys.path.insert(0, str(_parfact_dir))
        break

import parfact67 as legacy
import parfact67_scalar_streaming as scalar


FORMAT = "cinder67_scalar_curvature_v1"


def log(message: str) -> None:
    print(f"[cinder67-scalar] {message}", flush=True)


def config_json(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def in_memory_token_bank(path: Path, token_count: int, source_tokens: int,
                         context: int, offset: int) -> torch.Tensor:
    if token_count % context:
        raise ValueError("--fit-tokens must be divisible by --context")
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    if source_tokens < token_count:
        raise ValueError("--source-tokens must be at least --fit-tokens")
    if offset + source_tokens > len(raw):
        raise ValueError("token file is too short for the requested source span")
    # The copy is deliberate: all curvature probes reuse RAM, not the mmap.
    blocks = token_count // context
    source_blocks = source_tokens // context
    chosen = ((np.arange(blocks, dtype=np.int64) * source_blocks
               + source_blocks // 2) // blocks)
    values = np.empty((blocks, context), dtype=np.int64)
    for destination, block in enumerate(chosen):
        first = offset + int(block) * context
        values[destination] = raw[first:first + context]
    del raw
    bank = torch.from_numpy(values)
    if torch.cuda.is_available():
        bank = bank.pin_memory()
    return bank


def batches(bank: torch.Tensor, batch_sequences: int):
    for first in range(0, bank.shape[0], batch_sequences):
        yield first, bank[first:first + batch_sequences]


@torch.no_grad()
def prediction_contrasts(model, bank: torch.Tensor, batch_sequences: int,
                         topk: int, device: str) -> torch.Tensor:
    chunks = []
    for _, ids_cpu in batches(bank, batch_sequences):
        ids = ids_cpu.to(device, non_blocking=True)
        logits = model(ids).float()
        chunks.append(logits.topk(topk, dim=-1).indices.cpu())
        del ids, logits
    return torch.cat(chunks)


def hash_signs(count: int, sketch: int, seed: int, device: str,
               offset: int = 0) -> torch.Tensor:
    """Deterministic Rademacher event signs without persistent random state."""
    index = torch.arange(offset, offset + count, dtype=torch.int64,
                         device=device)
    value = index + 0x9E3779B9 * (sketch + 1) + 0x85EBCA6B * (seed + 1)
    value = (value ^ (value >> 16)) * 0x45D9F3B
    value = value ^ (value >> 16)
    return ((value & 1).float() * 2 - 1)


@torch.no_grad()
def parameter_probe(layout: scalar.ScalarLayout, probe: int, seed: int,
                    device: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed + 104729 * probe)
    result = {}
    for path, shape in layout.shapes.items():
        bits = torch.randint(0, 2, shape, generator=generator, device=device,
                             dtype=torch.int8)
        result[path] = bits.float().mul_(2).sub_(1)
    return result


@torch.no_grad()
def set_perturbation(model, trained: dict[str, torch.Tensor],
                     probe: dict[str, torch.Tensor], scale: float) -> None:
    for path in legacy.INTERNAL:
        model.get_submodule(path).weight.copy_(
            trained[path] * (1 + scale * probe[path]))


def sketched_gradients(model, parameters: list[torch.Tensor], ids: torch.Tensor,
                       contrast_ids: torch.Tensor, event_sketches: int,
                       global_first: int, total_events: int, seed: int
                       ) -> tuple[torch.Tensor, ...]:
    """Return one full scalar gradient per event sketch in one batched VJP."""
    logits = model(ids).float()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_contrast = contrast_ids.to(ids.device, non_blocking=True).reshape(
        -1, contrast_ids.shape[-1])
    predicted = flat_logits.gather(1, flat_contrast[:, :1]).squeeze(1)
    losses = []
    count = flat_logits.shape[0]
    for sketch in range(event_sketches):
        alternative = flat_logits.gather(
            1, flat_contrast[:, 1 + sketch:2 + sketch]).squeeze(1)
        # Global indices make separately batched positions one stable sign
        # sketch of the complete in-memory event bank.
        signs = hash_signs(count, sketch, seed, ids.device, global_first)
        losses.append(((predicted - alternative) * signs).sum() / total_events)
    loss_vector = torch.stack(losses)
    seeds = torch.eye(event_sketches, device=ids.device,
                      dtype=loss_vector.dtype)
    gradients = torch.autograd.grad(
        loss_vector, parameters, grad_outputs=seeds, is_grads_batched=True)
    del logits, flat_logits, flat_contrast, predicted, losses, loss_vector, seeds
    return gradients


def collect_probe(model, trained: dict[str, torch.Tensor],
                  layout: scalar.ScalarLayout, bank: torch.Tensor,
                  contrasts: torch.Tensor, probe_number: int,
                  event_sketches: int, batch_sequences: int, rho: float,
                  seed: int, device: str) -> dict[str, torch.Tensor]:
    probe = parameter_probe(layout, probe_number, seed, device)
    parameters = [model.get_submodule(path).weight for path in layout.paths]
    accumulator = {
        path: torch.zeros((event_sketches,) + layout.shapes[path],
                          device=device, dtype=torch.float32)
        for path in layout.paths
    }
    total_events = bank.numel()
    try:
        for first_sequence, ids_cpu in batches(bank, batch_sequences):
            ids = ids_cpu.to(device, non_blocking=True)
            target = contrasts[first_sequence:first_sequence + ids.shape[0]]
            set_perturbation(model, trained, probe, rho)
            plus = sketched_gradients(
                model, parameters, ids, target, event_sketches,
                first_sequence * bank.shape[1], total_events, seed)
            set_perturbation(model, trained, probe, -rho)
            minus = sketched_gradients(
                model, parameters, ids, target, event_sketches,
                first_sequence * bank.shape[1], total_events, seed)
            with torch.no_grad():
                for path, gp, gm in zip(layout.paths, plus, minus, strict=True):
                    # Convert weight gradients to conceptual gate gradients.
                    accumulator[path].add_(
                        (gp.float() - gm.float()) * trained[path][None],
                        alpha=1 / (2 * rho))
            del ids, target, plus, minus
    finally:
        set_perturbation(model, trained, probe, 0.0)
    return accumulator


def write_probe(features: np.memmap, layout: scalar.ScalarLayout,
                values: dict[str, torch.Tensor], first_column: int) -> None:
    width = next(iter(values.values())).shape[0]
    for index, path in enumerate(layout.paths):
        lo, hi = layout.offsets[index:index + 2]
        block = values[path].flatten(1).T.detach().cpu().numpy().astype(
            np.float16)
        features[lo:hi, first_column:first_column + width] = block
    features.flush()


@dataclass
class Segment:
    first: int
    last: int
    label: int


@torch.no_grad()
def prepare_features(features_path: Path, count: int, dimensions: int,
                     device: str) -> torch.Tensor:
    raw = np.memmap(features_path, dtype=np.float16, mode="r",
                    shape=(count, dimensions))
    x = torch.from_numpy(np.asarray(raw)).to(device)
    del raw
    # Global whitening removes arbitrary scale differences between probes.
    mean = x.float().mean(0)
    std = x.float().std(0, unbiased=False).clamp_min(1e-8)
    chunk = 1_000_000
    for first in range(0, count, chunk):
        block = (x[first:first + chunk].float() - mean) / std
        norm = block.square().sum(1, keepdim=True).sqrt().clamp_min(1e-6)
        x[first:first + chunk] = (block / norm).half()
    return x


@torch.no_grad()
def flattened_weight_mass(trained: dict[str, torch.Tensor],
                          layout: scalar.ScalarLayout,
                          device: str) -> torch.Tensor:
    return torch.cat([trained[path].flatten().float().square()
                      for path in layout.paths]).to(device)


@torch.no_grad()
def split_segment(x: torch.Tensor, mass: torch.Tensor, segment: Segment,
                  count_floor: float) -> tuple[torch.Tensor, int]:
    block = x[segment.first:segment.last]
    n = block.shape[0]
    center = block.float().mean(0)
    distance = (block.float() - center).square().sum(1)
    first_seed = int(distance.argmax())
    distance = (block.float() - block[first_seed].float()).square().sum(1)
    second_seed = int(distance.argmax())
    direction = block[first_seed].float() - block[second_seed].float()
    if float(direction.square().sum()) < 1e-12:
        direction = torch.ones(block.shape[1], device=block.device)
    score = block.float() @ direction
    order = score.argsort()

    ordered_mass = mass[segment.first:segment.last][order]
    cumulative = ordered_mass.cumsum(0)
    split = int(torch.searchsorted(cumulative, cumulative[-1] * .5)) + 1
    lower = max(1, int(math.floor(n * count_floor)))
    upper = min(n - 1, n - lower)
    split = max(lower, min(upper, split))
    return order, split


@torch.no_grad()
def balanced_partition(x: torch.Tensor, weight2: torch.Tensor,
                       components: int, count_floor: float
                       ) -> tuple[torch.Tensor, list[dict]]:
    if components & (components - 1):
        raise ValueError("components must be a power of two")
    count = x.shape[0]
    ids = torch.arange(count, dtype=torch.int32, device=x.device)
    # Half count mass and half Frobenius mass guards against both giant scalar
    # buckets and giant weight-mass buckets.
    mass = (torch.full_like(weight2, 1 / count) +
            weight2 / weight2.sum().clamp_min(1e-30)) * .5
    segments = [Segment(0, count, 0)]
    history = []
    for depth in range(int(math.log2(components))):
        new_x = torch.empty_like(x)
        new_ids = torch.empty_like(ids)
        new_mass = torch.empty_like(mass)
        children = []
        cursor = 0
        ratios = []
        for segment in segments:
            order, split = split_segment(x, mass, segment, count_floor)
            n = segment.last - segment.first
            ordered_x = x[segment.first:segment.last][order]
            ordered_ids = ids[segment.first:segment.last][order]
            ordered_mass = mass[segment.first:segment.last][order]
            new_x[cursor:cursor + n] = ordered_x
            new_ids[cursor:cursor + n] = ordered_ids
            new_mass[cursor:cursor + n] = ordered_mass
            label = 2 * segment.label
            children.append(Segment(cursor, cursor + split, label))
            children.append(Segment(cursor + split, cursor + n, label + 1))
            ratios.append(split / n)
            cursor += n
            del order, ordered_x, ordered_ids, ordered_mass
        x, ids, mass, segments = new_x, new_ids, new_mass, children
        row = {"depth": depth + 1, "groups": len(segments),
               "min_left_fraction": min(ratios),
               "max_left_fraction": max(ratios)}
        history.append(row)
        log(f"partition depth {depth+1}/{int(math.log2(components))}: "
            f"{len(segments):,} groups")

    owner = torch.empty(count, dtype=torch.int16, device=x.device)
    for segment in segments:
        owner[ids[segment.first:segment.last].long()] = segment.label
    return owner.cpu(), history


def owner_payload(owner: torch.Tensor, layout: scalar.ScalarLayout,
                  components: int) -> dict[str, torch.Tensor]:
    shift = int(math.log2(owner.max().item() + 1)) - int(math.log2(components))
    result = {}
    for index, path in enumerate(layout.paths):
        lo, hi = layout.offsets[index:index + 2]
        result[path] = (owner[lo:hi].long() >> shift).short().view(
            layout.shapes[path])
    return result


def run(args) -> None:
    if args.components & (args.components - 1):
        raise ValueError("--components must be a power of two")
    args.run.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    model = legacy.load_model(args.checkpoint, args.device, args.sensor).eval()
    layout = scalar.scalar_layout(model)
    trained = {path: model.get_submodule(path).weight.detach().clone()
               for path in layout.paths}
    dimensions = args.parameter_probes * args.event_sketches
    feature_path = args.run / "curvature_features.f16"
    expected_bytes = layout.count * dimensions * np.dtype(np.float16).itemsize
    if feature_path.exists() and feature_path.stat().st_size != expected_bytes:
        raise ValueError(f"unexpected feature file size: {feature_path}")
    mode = "r+" if feature_path.exists() else "w+"
    features = np.memmap(feature_path, dtype=np.float16, mode=mode,
                         shape=(layout.count, dimensions))
    progress_path = args.run / "collection.json"
    progress = (json.loads(progress_path.read_text()) if progress_path.exists()
                else {"format": FORMAT, "config": config_json(args),
                      "completed_probes": 0, "history": []})
    if progress["config"] != config_json(args):
        raise ValueError("resume configuration differs from collection metadata")

    bank = in_memory_token_bank(args.tokens, args.fit_tokens,
                                args.source_tokens, args.context,
                                args.token_offset)
    log(f"loaded {bank.numel():,} Pile positions, stratified over "
        f"{args.source_tokens:,}, into RAM once")
    contrasts = prediction_contrasts(
        model, bank, args.batch_sequences, args.event_sketches + 1,
        args.device)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1, request_stop)
    started = time.perf_counter()
    for probe_number in range(progress["completed_probes"],
                              args.parameter_probes):
        probe_started = time.perf_counter()
        values = collect_probe(
            model, trained, layout, bank, contrasts, probe_number,
            args.event_sketches, args.batch_sequences, args.rho, args.seed,
            args.device)
        write_probe(features, layout, values,
                    probe_number * args.event_sketches)
        seconds = time.perf_counter() - probe_started
        progress["completed_probes"] = probe_number + 1
        progress["history"].append({"probe": probe_number,
                                    "seconds": seconds})
        atomic_json(progress, progress_path)
        log(f"collected probe {probe_number+1}/{args.parameter_probes} "
            f"in {seconds:.1f}s")
        del values
        if stop_requested:
            log("stopping cleanly after durable probe checkpoint")
            return

    del contrasts, bank, model
    torch.cuda.empty_cache()
    x = prepare_features(feature_path, layout.count, dimensions, args.device)
    weight2 = flattened_weight_mass(trained, layout, args.device)
    owner, partition_history = balanced_partition(
        x, weight2, args.components, args.count_floor)
    del x, weight2
    torch.cuda.empty_cache()

    owners = owner_payload(owner, layout, args.components)
    artifact = {
        "format": FORMAT, "components": args.components,
        "atoms": layout.count, "atom_basis": "individual scalar parameters",
        "scope": list(layout.paths), "shapes": layout.shapes,
        "owner": owners, "background_component": False,
        "exclusive_hard_ownership": True, "exact_reconstruction": True,
        "nested_components": [c for c in (1024, 2048, 4096)
                              if c <= args.components],
        "feature_dimensions": dimensions,
    }
    legacy.atomic_torch_save(artifact, args.run / "partition.pt")
    manifest = {
        "format": FORMAT, "complete": True,
        "parameter_scope": "goodfire_vpd_24_internal_linear_matrices",
        "atoms": layout.count, "atom_basis": "individual scalar parameters",
        "components": args.components,
        "nested_components": artifact["nested_components"],
        "ownership": "one hard owner per scalar; no background",
        "curvature_estimator": "central finite transport of gate gradients",
        "feature_dimensions": dimensions,
        "fit_tokens": args.fit_tokens,
        "data_path": str(args.tokens),
        "data_residency": "copied once to host RAM; reused by every probe",
        "random_ablation_training": False,
        "svd_atoms": False, "pca": False, "centroids": False,
        "partition_history": partition_history,
        "elapsed_seconds": time.perf_counter() - started,
        "config": config_json(args),
    }
    atomic_json(manifest, args.run / "manifest.json")
    log(f"complete: {layout.count:,} scalar atoms, C={args.components}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", type=Path, required=True)
    result.add_argument("--tokens", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--sensor", choices=("grad", "gim"), default="grad")
    result.add_argument("--fit-tokens", type=int, default=65_536)
    result.add_argument("--source-tokens", type=int, default=100_000_000)
    result.add_argument("--token-offset", type=int, default=0)
    result.add_argument("--context", type=int, default=512)
    result.add_argument("--batch-sequences", type=int, default=4)
    result.add_argument("--parameter-probes", type=int, default=8)
    result.add_argument("--event-sketches", type=int, default=4)
    result.add_argument("--rho", type=float, default=0.02)
    result.add_argument("--components", type=int, default=4096)
    result.add_argument("--count-floor", type=float, default=1 / 3)
    result.add_argument("--seed", type=int, default=314159)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
