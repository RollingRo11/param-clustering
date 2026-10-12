"""Native-sensor, one-component-at-a-time ablation of the two 50k toy fits."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch

import parauto_core as core
import parauto_structured as structured
import parauto_toy as toy
import parauto_tracking as tracking
from parauto_evaluate import TARGET_SHA, all_scores
from parauto_io import load_pieces, method_directory
from parauto_target import atomic_json, atomic_save, fingerprint, load_target, sample


def removal_ranks(scores):
    """Global per-example ascending order; exact ties use matrix/component index."""
    flat = torch.cat(scores, -1)
    if not torch.isfinite(flat).all():
        raise ValueError("Nonfinite importance score")
    order = flat.argsort(dim=-1, stable=True)
    return torch.empty_like(order).scatter_(-1, order, torch.arange(
        flat.shape[-1], device=flat.device).expand_as(order))


def keep_masks(ranks, removed, counts):
    """Return masks [number of removal levels, examples, components] per matrix."""
    return (ranks.unsqueeze(0) >= removed[:, None, None]).split(counts, dim=-1)


def distortions(reference, candidate, temperature=1.0):
    """KL(softmax(reference/T) || softmax(candidate/T)), plus raw-output MSE."""
    logp = (reference.double() / temperature).log_softmax(-1)
    logq = (candidate.double() / temperature).log_softmax(-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)
    if not torch.isfinite(kl).all() or float(kl.min()) < -1e-10:
        raise ValueError("Invalid KL")
    return kl.clamp_min(0), (candidate.double() - reference.double()).square().mean(-1)


@torch.no_grad()
def evaluate_curve(target, x, pieces, ranks, *, batch, level_batch, temperature):
    """Every level uses a fresh nonlinear forward, with frozen original rankings."""
    total = sum(p.count for p in pieces)
    counts = [p.count for p in pieces]
    values = {key: torch.empty(len(x), total+1, dtype=torch.float64)
              for key in ("kl_target", "mse_target", "kl_own_full", "mse_own_full")}
    full = target(x)
    own_full = toy.forward_with_pieces(target, x, pieces,
        [torch.ones(len(x), c, device=x.device) for c in counts])[0]
    started = time.perf_counter()
    for start in range(0, len(x), batch):
        xb = x[start:start+batch]
        size = len(xb)
        for first in range(0, total+1, level_batch):
            removed = torch.arange(first, min(first+level_batch, total+1), device=x.device)
            masks = [m.reshape(-1, c).to(x.dtype) for m, c in zip(
                keep_masks(ranks[start:start+size], removed, counts), counts, strict=True)]
            repeated = xb.unsqueeze(0).expand(len(removed), -1, -1).reshape(-1, xb.shape[-1])
            y = toy.forward_with_pieces(target, repeated, pieces, masks)[0].reshape(len(removed), size, -1)
            kl, mse = distortions(full[start:start+size], y, temperature)
            own_kl, own_mse = distortions(own_full[start:start+size], y, temperature)
            for key, value in zip(values, (kl, mse, own_kl, own_mse), strict=True):
                values[key][start:start+size, first:first+len(removed)] = value.T.cpu()
        print(json.dumps({"evaluated_examples": start+size, "total_examples": len(x),
                          "elapsed_seconds": time.perf_counter()-started}), flush=True)
        tracking.log_metrics({'evaluated_examples': start+size, 'total_examples': len(x),
                              'elapsed_seconds': time.perf_counter()-started}, prefix='evaluation')
    # Stored columns follow actual removal order: 0 through C_total.
    zero = x @ target.W_E @ target.W_U
    expected_zero = distortions(full, zero, temperature)
    for key, expected in zip(("kl_target", "mse_target"), expected_zero, strict=True):
        torch.testing.assert_close(values[key][:, -1], expected.cpu(), atol=1e-10, rtol=2e-5)
    torch.testing.assert_close(values["mse_own_full"][:, 0], torch.zeros(len(x), dtype=torch.float64),
                               atol=1e-10, rtol=0)
    return values, own_full


def summarize(values, meaningful, counts, scores):
    total = sum(counts)
    out = {"removed": list(range(total+1)), "kept": list(range(total, -1, -1)),
           "components_per_matrix": counts, "metrics": {}}
    for name, data in values.items():
        out["metrics"][name] = {"mean": data.mean(0).tolist(),
            "standard_error": (data.std(0) / len(data)**0.5).tolist(),
            "mean_meaningful": data[meaningful].mean(0).tolist(),
            "p95_meaningful": data[meaningful].quantile(.95, dim=0).tolist()}
    ranks = removal_ranks(scores)
    out["selected_budgets"] = {}
    for kept in (0, 1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 800, total):
        index = total-kept
        masks = keep_masks(ranks, torch.tensor([index], device=ranks.device), counts)
        out["selected_budgets"][str(kept)] = {
            **{name: float(data[:, index].mean()) for name, data in values.items()},
            "kept_per_matrix": [float(m.float().mean(1).sum()) for m in masks]}
    # Do not assume monotonicity: record both the first passing budget and the
    # first budget for which *all* larger budgets also pass.
    out["normalized_target_thresholds"] = {}
    for name in ("kl_target", "mse_target"):
        average = values[name].mean(0).flip(0)
        normalized = average / average[0].clamp_min(1e-30)
        worst_suffix = normalized.flip(0).cummax(0).values.flip(0)
        out["normalized_target_thresholds"][name] = {}
        for fraction in (.01, .05, .1):
            good = (normalized <= fraction).nonzero().flatten()
            sustained = (worst_suffix <= fraction).nonzero().flatten()
            out["normalized_target_thresholds"][name][str(fraction)] = {
                "first_kept": int(good[0]) if len(good) else None,
                "all_larger_budgets_pass_from": int(sustained[0]) if len(sustained) else None}
    return out


def draw(result, root):
    colors = {"par_auto": "#1976b9", "spd": "#df7426"}
    labels = {"par_auto": "Par-auto: output-effect sensor", "spd": "SPD: learned causal importance"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (metric, title) in enumerate((("kl_target", "KL after softmax (T=1)"),
                                         ("mse_target", "Raw-output MSE"))):
        for column, limit in enumerate((1600, 64)):
            ax = axes[row, column]
            for name, entry in result["methods"].items():
                ax.plot(entry["kept"][:-1], entry["metrics"][metric]["mean"][:-1], color=colors[name], label=labels[name])
            ax.set(xlim=(1, limit), xlabel="Components retained across all four matrices",
                   ylabel=title, title="Full range · log x axis" if column == 0 else "Sparse-budget detail")
            ax.set_xscale("log")
            ax.set_xticks([1, 4, 16, 64, 256, 1600] if column == 0 else [1, 2, 4, 8, 16, 32, 64])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_yscale("symlog", linthresh=1e-10)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Least-importance-first ablation · 50k fits · 400 components/matrix\n"
                 f"{result['events']:,} fresh shared inputs · frozen 10k-step target · lower is better")
    for suffix in ("png", "svg"):
        fig.savefig(root / f"klkeep.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("kl_target", "mse_target"), strict=True):
        for name, entry in result["methods"].items():
            mean = torch.tensor(entry["metrics"][metric]["mean"])
            ax.plot(entry["removed"][1:], (mean / mean[-1])[1:], color=colors[name], label=labels[name])
        ax.set(xlabel="Components removed, least → most important", ylabel=f"{metric} / all-removed error",
               xlim=(1, 1600))
        ax.set_xscale("log")
        ax.set_xticks([1, 4, 16, 64, 256, 1600])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Cumulative ablation in each method's frozen sensor order")
    fig.savefig(root / "removal_curve.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric, label in zip(axes, ("kl_own_full", "mse_own_full"),
                                ("KL vs own all-kept output", "MSE vs own all-kept output"), strict=True):
        for name, entry in result["methods"].items():
            ax.plot(entry["kept"][:-1], entry["metrics"][metric]["mean"][:-1], color=colors[name], label=labels[name])
        ax.set(xlabel="Components retained across all four matrices", ylabel=label, xlim=(1, 1600))
        ax.set_xscale("log")
        ax.set_xticks([1, 4, 16, 64, 256, 1600])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Removal damage relative to each decomposition's own full output")
    fig.savefig(root / "own_reference.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def main():
    from benchmark_spd import author_target, load_author_model
    from spd.configs import Config
    from spd.models.component_utils import calc_causal_importances
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--method-run", type=Path, required=True)
    parser.add_argument("--spd-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--level-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=920091)
    args = parser.parse_args()
    tracking.start_cli(args)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if fingerprint(args.target) != TARGET_SHA:
        raise ValueError("Expected the verified 10k-step target")
    method_run = method_directory(args.method_run)
    for run in (method_run, args.spd_run):
        meta = json.loads((run / "result.json").read_text())
        if (meta["steps_per_matrix"], meta["components_per_matrix"], meta["target_sha256"]) != (50000, 400, TARGET_SHA):
            raise ValueError(f"Wrong trained run: {run}")
    target = load_target(args.target, "cuda")
    x = sample(args.events, 100, torch.Generator(device="cuda").manual_seed(args.seed), "cuda")
    full = target(x)
    meaningful = ((full - x @ target.W_E @ target.W_U).square().mean(-1) > toy.EPSILON).cpu()
    if not meaningful.any():
        raise ValueError("No meaningful examples")
    cfg = Config(**json.loads((args.spd_run / "executed_config.json").read_text()))
    author = load_author_model(author_target(args.target), cfg, args.spd_run / "model_50000.pth")
    torch.testing.assert_close(author.model(x), full, atol=1e-6, rtol=1e-5)
    components = {k.replace("-", "."): v for k, v in author.components.items()}
    gates = {k.replace("-", "."): v for k, v in author.gates.items()}
    names = [f"layers.{i}.{kind}" for i in range(2) for kind in ("mlp_in", "mlp_out")]
    spd_pieces = [core.FreePieces(components[n].A.detach(), components[n].B.T.detach()) for n in names]
    _, pre = author.forward_with_pre_forward_cache_hooks(x, module_names=names)
    ci, _ = calc_causal_importances(pre, {n: components[n].A for n in names}, gates)
    pieces = load_pieces(method_run, target, "cuda")
    cache = structured.cache_events(target, x)
    scores = all_scores(cache, pieces, target)
    del cache
    result = {"format": "toy_native_sensor_klkeep_v1", "events": len(x),
        "meaningful_events": int(meaningful.sum()), "seed": args.seed, "target_sha256": TARGET_SHA,
        "target_steps": 10000, "decomposition_steps": 50000, "temperature": 1.0,
        "ranking": "per-example global ranking of 1600 independent matrix components, frozen on original target activations",
        "tie_break": "stable ascending matrix/component index; no reranking after removal",
        "masks": "binary retained/removed; no continuous CI rescaling and no undecomposed weight residual",
        "kl_definition": "KL(softmax(frozen target output) || softmax(masked decomposition output)), temperature 1",
        "metric_caveat": "Toy outputs are regression values, not native logits; raw-output MSE is also reported",
        "checkpoints": {"par_auto": str(method_run / "pieces.pt"), "spd": str(args.spd_run / "model_50000.pth")},
        "checkpoint_sha256": {"par_auto": fingerprint(method_run / "pieces.pt"), "spd": fingerprint(args.spd_run / "model_50000.pth")},
        "methods": {}}
    for name, parts, importance in (("par_auto", pieces, scores), ("spd", spd_pieces, [ci[n] for n in names])):
        print(f"START {name}", flush=True)
        ranks = removal_ranks(importance)
        if name == "spd":
            # Check the shared evaluator against the original authors' forward
            # for full, empty, and intermediate globally ranked binary masks.
            for removed in (0, 800, 1600):
                masks = [m[0].float() for m in keep_masks(ranks[:32], torch.tensor([removed], device="cuda"), [400]*4)]
                expected = author.forward_with_components(x[:32], components=components, masks=dict(zip(names, masks, strict=True)))
                actual = toy.forward_with_pieces(target, x[:32], parts, masks)[0]
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
            result["spd_forward_parity"] = True
        values, own_full = evaluate_curve(target, x, parts, ranks, batch=args.batch,
                                         level_batch=args.level_batch, temperature=1.0)
        if name == "par_auto":
            torch.testing.assert_close(own_full, full, atol=2e-5, rtol=1e-5)
        result["methods"][name] = summarize(values, meaningful, [p.count for p in parts], importance)
        atomic_save({"x": x.cpu(), "meaningful": meaningful, "removal_ranks": ranks.cpu(),
                     "scores": [s.cpu() for s in importance], **values}, args.output / f"{name}_events.pt")
        atomic_json(result, args.output / "result.json")
    draw(result, args.output)
    print("KLKEEP_COMPLETE", flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
