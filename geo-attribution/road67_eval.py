"""Held-out KLKeep and component-mass evaluation for ROAD-67."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# Parfact implementations live in the sibling parfact directory.
import sys as _parfact_sys
from pathlib import Path as _ParfactPath
for _parfact_dir in (_ParfactPath(__file__).resolve().parent / "parfact",
                     _ParfactPath(__file__).resolve().parent.parent / "parfact"):
    if _parfact_dir.is_dir():
        _parfact_sys.path.insert(0, str(_parfact_dir))
        break

import parfact67 as legacy
import parfact67_eval as old
import parfact67_streaming as stream
import plot_eval_svg
import road67


def log(message: str) -> None:
    print(f"[road67-eval] {message}", flush=True)


@torch.no_grad()
def hard_mask(logits: torch.Tensor, keep: int) -> torch.Tensor:
    if keep == 0:
        return torch.zeros_like(logits)
    if keep == logits.shape[-1]:
        return torch.ones_like(logits)
    indices = logits.topk(keep, dim=-1).indices
    return torch.zeros_like(logits).scatter_(-1, indices, 1)


@torch.no_grad()
def mass_stats(system: road67.RoadSystem, components: int) -> dict:
    device = next(system.parameters()).device
    mass = torch.zeros(components, dtype=torch.float64, device=device)
    counts = torch.zeros(components, dtype=torch.int64, device=device)
    for module in system.routed.values():
        left = module.rotation(module.left_factor)
        right = module.rotation(module.right_factor)
        shard_mass = (left.float().square().sum(0)
                      * right.float().square().sum(0)).double()
        owner = module.hard_owner()
        mass.scatter_add_(0, owner, shard_mass)
        counts.scatter_add_(0, owner, torch.ones_like(owner))
    share = (mass / mass.sum()).cpu()
    ordered = share.sort().values
    index = torch.arange(1, components + 1, dtype=torch.float64)
    gini = float(((2 * index - components - 1) * ordered).sum()
                 / (components * ordered.sum()).clamp_min(1e-30))
    mean, centered = share.mean(), share - share.mean()
    entropy = float(-(share * share.clamp_min(1e-30).log()).sum())
    descending = share.sort(descending=True).values
    top_values, top_ids = share.topk(min(10, components))
    return {
        "definition": "sum of rotated rank-one shard Frobenius-squared mass",
        "background_share": 0.0, "candidate_share": 1.0,
        "smallest_component_shards": int(counts.min()),
        "largest_component_shards": int(counts.max()),
        "dead_component_count": int((counts == 0).sum()),
        "largest_component_share_of_total": float(descending[0]),
        "largest_over_uniform_candidate": float(share.max() * components),
        "coefficient_of_variation": float(share.std(unbiased=False) / mean),
        "raw_skewness": float(centered.pow(3).mean()
                              / centered.pow(2).mean().pow(1.5).clamp_min(1e-30)),
        "gini": gini, "effective_components_entropy": math.exp(entropy),
        "top10_share_of_total": float(descending[:10].sum()),
        "top1pct_share_of_total": float(
            descending[:max(1, math.ceil(components * .01))].sum()),
        "top10pct_share_of_total": float(
            descending[:max(1, math.ceil(components * .10))].sum()),
        "top10_components": [
            {"component": int(component), "share_of_total": float(value)}
            for value, component in zip(top_values, top_ids, strict=True)],
        "component_shares_of_total_sorted": descending.tolist(),
    }


def run(args) -> None:
    artifact_path = args.run / "road.pt"
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu",
                              weights_only=False)
    else:
        checkpoint_path = args.run / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
        config = checkpoint["config"]
        artifact = {
            "format": checkpoint["format"], "config": config,
            "components": int(config["components"]),
            "tokens": int(checkpoint["tokens"]),
            "trainable": checkpoint["trainable"],
        }
        log(f"evaluating interim checkpoint at {artifact['tokens']:,} tokens")
    if artifact.get("format") != road67.FORMAT:
        raise ValueError("incompatible ROAD artifact")
    components = int(artifact["components"])
    capability = torch.cuda.get_device_capability(torch.device(args.device))
    autocast_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    target = legacy.load_model(args.checkpoint, args.device, "grad").eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    basis = stream.load_basis(args.basis, args.device)
    config = artifact["config"]
    system = road67.RoadSystem(
        target, basis, components, int(config["transforms"]),
        int(config["router_width"]), int(config["seed"])).to(args.device).eval()
    system.load_trainable_payload(artifact["trainable"])
    mass = mass_stats(system, components)
    log(f"mass: largest={100*mass['largest_component_share_of_total']:.3f}% "
        f"dead={mass['dead_component_count']}")

    ids_cpu, positions = old.heldout_events(
        args.tokens, args.target_tokens, args.seq, args.events, args.seed)
    keep_grid = old.grid(components)
    values = np.zeros((args.events, len(keep_grid)), dtype=np.float64)
    started = time.perf_counter()
    for event in range(args.events):
        position = int(positions[event])
        ids = ids_cpu[event, :position + 1][None].to(args.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
            states, target_logits = road67.target_states(target, ids, True)
            router_logits = system.router(states[:-1])
        target_logp = F.log_softmax(target_logits[0, position].float(), -1)
        for index, keep in enumerate(keep_grid):
            mask = hard_mask(router_logits, keep)
            with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
                observed = system.forward_states(ids, mask)
                logits = system.student.lm_head(observed[-1])[0, position].float()
            values[event, index] = float(F.kl_div(
                F.log_softmax(logits, -1), target_logp, log_target=True,
                reduction="sum"))
        if (event + 1) % 4 == 0:
            log(f"KLKeep {event+1}/{args.events}")

    klkeep = {
        str(components): {
            "keep_grid": keep_grid, "kl_mean": values.mean(0).tolist(),
            "kl_median": np.median(values, 0).tolist(),
            "kl_p25": np.quantile(values, .25, axis=0).tolist(),
            "kl_p75": np.quantile(values, .75, axis=0).tolist(),
            "kl_p90": np.quantile(values, .90, axis=0).tolist(),
            "thresholds": old.threshold_summary(values, keep_grid),
            "roundtrip_kl_at_c": float(values[:, -1].mean()),
        }
    }
    result = {
        "format": "road67_eval_v1", "method": "ROAD",
        "training_tokens": artifact.get("tokens"),
        "events": args.events, "positions": positions.tolist(),
        "sensor": "learned cheap residual router",
        "attribution_order": "per-position ROAD router score",
        "kl_definition": "KL(target || top-k routed global components)",
        "background_component": False, "clusters": [components],
        "klkeep": klkeep, "weight_mass": {str(components): mass},
        "elapsed_seconds": time.perf_counter() - started,
    }
    out_dir = args.run / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "stats.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    plot_eval_svg.plot(result, out_dir)
    log(f"wrote {output} and plots")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", type=Path, required=True)
    result.add_argument("--tokens", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--basis", type=Path, required=True)
    result.add_argument("--target-tokens", type=int, default=100_000_000)
    result.add_argument("--events", type=int, default=24)
    result.add_argument("--seq", type=int, default=512)
    result.add_argument("--seed", type=int, default=8675309)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
