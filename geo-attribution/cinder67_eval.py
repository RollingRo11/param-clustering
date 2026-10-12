"""KLKeep and weight-mass evaluation for hard scalar CINDER partitions."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cinder67_scalar as cinder

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
import plot_eval_svg


def log(message: str) -> None:
    print(f"[cinder67-eval] {message}", flush=True)


@torch.no_grad()
def owners_at(owner4096: dict[str, torch.Tensor], maximum: int,
              components: int, device: str) -> dict[str, torch.Tensor]:
    shift = int(math.log2(maximum)) - int(math.log2(components))
    return {path: (value.to(device).long() >> shift)
            for path, value in owner4096.items()}


@torch.no_grad()
def component_scores(attribution: dict[str, torch.Tensor],
                     owner: dict[str, torch.Tensor], components: int,
                     device: str) -> torch.Tensor:
    result = torch.zeros(components, device=device)
    for path in legacy.INTERNAL:
        result.scatter_add_(0, owner[path].flatten(),
                            attribution[path].flatten())
    return result


@torch.no_grad()
def apply_keep(model, trained: dict[str, torch.Tensor],
               owner: dict[str, torch.Tensor], rank: torch.Tensor,
               keep: int) -> None:
    for path in legacy.INTERNAL:
        selected = rank[owner[path]] < keep
        model.get_submodule(path).weight.copy_(trained[path] * selected)


@torch.no_grad()
def mass_stats(owner: dict[str, torch.Tensor],
               trained: dict[str, torch.Tensor], components: int,
               device: str) -> tuple[dict, torch.Tensor]:
    mass = torch.zeros(components, dtype=torch.float64, device=device)
    total = torch.zeros((), dtype=torch.float64, device=device)
    counts = torch.zeros(components, dtype=torch.int64, device=device)
    for path in legacy.INTERNAL:
        weight2 = trained[path].double().square()
        flat_owner = owner[path].flatten()
        total += weight2.sum()
        mass.scatter_add_(0, flat_owner, weight2.flatten())
        counts.scatter_add_(0, flat_owner, torch.ones_like(flat_owner))
    share = (mass / total).cpu()
    normalized = share / share.sum().clamp_min(1e-30)
    ordered = normalized.sort().values
    index = torch.arange(1, components + 1, dtype=torch.float64)
    gini = float(((2 * index - components - 1) * ordered).sum() /
                 (components * ordered.sum()).clamp_min(1e-30))
    mean = normalized.mean()
    centered = normalized - mean
    entropy = float(-(normalized * normalized.clamp_min(1e-30).log()).sum())
    descending = share.sort(descending=True).values
    top_values, top_ids = share.topk(min(10, components))
    result = {
        "definition": "sum of squared scalar weights under exclusive ownership",
        "total_frobenius_squared": float(total),
        "background_share": 0.0, "candidate_share": 1.0,
        "smallest_component_parameters": int(counts.min()),
        "largest_component_parameters": int(counts.max()),
        "largest_component_share_of_total": float(descending[0]),
        "largest_over_uniform_candidate": float(normalized.max() * components),
        "coefficient_of_variation": float(normalized.std(unbiased=False) / mean),
        "raw_skewness": float(centered.pow(3).mean() /
                              centered.pow(2).mean().pow(1.5).clamp_min(1e-30)),
        "gini": gini,
        "effective_components_entropy": math.exp(entropy),
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
    return result, share


def run(args) -> None:
    artifact = torch.load(args.run / "partition.pt", map_location="cpu",
                          weights_only=False)
    if artifact.get("format") != cinder.FORMAT:
        raise ValueError("incompatible scalar CINDER partition")
    maximum = int(artifact["components"])
    clusters = [int(value) for value in args.clusters.split(",")]
    if any(c > maximum or c & (c - 1) for c in clusters):
        raise ValueError("requested C must be a power of two no larger than partition")

    model = legacy.load_model(args.checkpoint, args.device, "gim").eval()
    trained = {path: model.get_submodule(path).weight.detach().clone()
               for path in legacy.INTERNAL}
    owners = {c: owners_at(artifact["owner"], maximum, c, args.device)
              for c in clusters}
    masses, mass_arrays = {}, {}
    for c in clusters:
        masses[str(c)], mass_arrays[c] = mass_stats(
            owners[c], trained, c, args.device)
        log(f"C={c}: largest mass={100*masses[str(c)]['largest_component_share_of_total']:.3f}%")

    ids_cpu, positions = old.heldout_events(
        args.tokens, args.target_tokens, args.seq, args.events, args.seed)
    orders = {c: [] for c in clusters}
    base_logprobs = []
    started = time.perf_counter()
    for event in range(args.events):
        ids = ids_cpu[event].to(args.device)
        attribution, base = old.token_attribution(
            model, trained, ids, int(positions[event]))
        base_logprobs.append(base.cpu())
        for c in clusters:
            scores = component_scores(attribution, owners[c], c, args.device)
            orders[c].append(scores.abs().argsort(descending=True).cpu())
        if (event + 1) % 4 == 0:
            log(f"ranked {event+1}/{args.events} predictions")

    klkeep = {}
    for c in clusters:
        keep_grid = old.grid(c)
        values = np.zeros((args.events, len(keep_grid)), dtype=np.float64)
        for event in range(args.events):
            order = orders[c][event].to(args.device)
            rank = torch.empty(c, dtype=torch.int16, device=args.device)
            rank[order] = torch.arange(c, dtype=torch.int16, device=args.device)
            ids = ids_cpu[event].to(args.device)
            position = int(positions[event])
            for index, keep in enumerate(keep_grid):
                apply_keep(model, trained, owners[c], rank, keep)
                with torch.no_grad():
                    logits = model(ids[:position + 1][None])[0, position].float()
                    values[event, index] = float(F.kl_div(
                        F.log_softmax(logits, -1),
                        base_logprobs[event].to(args.device), log_target=True,
                        reduction="sum"))
            old.restore({path: model.get_submodule(path).weight
                         for path in legacy.INTERNAL}, trained)
            if (event + 1) % 4 == 0:
                log(f"C={c} KLKeep {event+1}/{args.events}")
        klkeep[str(c)] = {
            "keep_grid": keep_grid, "kl_mean": values.mean(0).tolist(),
            "kl_median": np.median(values, 0).tolist(),
            "kl_p25": np.quantile(values, .25, axis=0).tolist(),
            "kl_p75": np.quantile(values, .75, axis=0).tolist(),
            "kl_p90": np.quantile(values, .90, axis=0).tolist(),
            "thresholds": old.threshold_summary(values, keep_grid),
            "roundtrip_kl_at_c": float(values[:, -1].mean()),
        }

    result = {
        "format": "cinder67_scalar_eval_v1", "method": "Scalar CINDER",
        "events": args.events, "positions": positions.tolist(),
        "sensor": "gim",
        "attribution_order": "absolute signed-sum scalar GIM grad*weight attribution to gold next-token logit",
        "kl_definition": "KL(full model || top-k exclusively owned scalar components)",
        "background_component": False, "clusters": clusters,
        "klkeep": klkeep, "weight_mass": masses,
        "elapsed_seconds": time.perf_counter() - started,
    }
    out_dir = args.run / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "stats.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    plot_eval_svg.plot(result, out_dir)
    log(f"wrote {output} and KLKeep/mass plots")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run", type=Path, required=True)
    result.add_argument("--tokens", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--target-tokens", type=int, default=100_000_000)
    result.add_argument("--clusters", default="1024,2048,4096")
    result.add_argument("--events", type=int, default=24)
    result.add_argument("--seq", type=int, default=512)
    result.add_argument("--seed", type=int, default=8675309)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
