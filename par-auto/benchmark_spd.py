"""Thin evaluation/timing harness around the authors' SPD and our projected Lp method.

No SPD loss, gate, factorization, or optimizer update is reimplemented here.
Use the unmodified spd-paper source at c6314c9 on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_method as objectives  # noqa: E402
import parauto_toy as toy  # noqa: E402
import plot_spd_comparison as comparison  # noqa: E402
import spd.run_spd as upstream  # noqa: E402
from parauto_evaluate import TARGET_SHA  # noqa: E402
from parauto_target import (  # noqa: E402  # noqa: E402
    atomic_json,
    atomic_save,
    fingerprint,
    load_target,
    sample,
)
from spd.configs import Config  # noqa: E402
from spd.data_utils import DatasetGeneratedDataLoader  # noqa: E402
from spd.experiments.resid_mlp.models import (  # noqa: E402
    ResidualMLP,
    ResidualMLPConfig,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset  # noqa: E402
from spd.experiments.resid_mlp.resid_mlp_interp import (  # noqa: E402
    compute_spd_weight_neuron_contributions,
    compute_target_weight_neuron_contributions,
    plot_neuron_contribution_pairs,
)
from spd.models.component_model import ComponentModel  # noqa: E402
from spd.models.component_utils import calc_causal_importances  # noqa: E402
from spd.utils import set_seed  # noqa: E402


class QuietProgress:
    """Only replace tqdm display; the upstream training loop stays unchanged."""
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    @staticmethod
    def write(message):
        print(message, flush=True)


class Meter:
    """Synchronized wall time, PyTorch peaks, and sampled NVIDIA telemetry."""
    def __init__(self, root, label):
        self.path = root / f"gpu_{label}.csv"

    def __enter__(self):
        self.stream = self.path.open("w")
        self.proc = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits", "--loop-ms=200"], stdout=self.stream, stderr=subprocess.DEVNULL)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.start_unix = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.start
        end = time.time()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.stream.close()
        samples = []
        for line in self.path.read_text().splitlines():
            try:
                samples.append([float(x.strip()) for x in line.split(",")[1:]])
            except ValueError:
                continue
        data = np.asarray(samples)
        self.result = {"wall_seconds": seconds, "start_unix": self.start_unix, "end_unix": end,
                       "peak_torch_allocated_bytes": allocated, "peak_torch_reserved_bytes": reserved,
                       "telemetry_samples": len(samples),
                       "mean_gpu_utilization_pct": float(data[:, 0].mean()) if len(samples) else None,
                       "mean_memory_controller_utilization_pct": float(data[:, 1].mean()) if len(samples) else None,
                       "peak_nvidia_memory_used_mib": float(data[:, 2].max()) if len(samples) else None,
                       "mean_power_watts": float(data[:, 3].mean()) if len(samples) else None}


def author_target(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ResidualMLPConfig(**state["config"], act_fn_name="relu", in_bias=False, out_bias=False)
    target = ResidualMLP(cfg)
    target.load_state_dict(state["model"], strict=True)
    return target.cuda().eval().requires_grad_(False)


def load_author_model(target, cfg, checkpoint):
    model = ComponentModel(target, cfg.target_module_patterns, cfg.C, cfg.n_ci_mlp_neurons,
                           cfg.pretrained_model_output_attr).cuda()
    expected = {k: v.detach().clone() for k, v in target.state_dict().items()}
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
    for name, value in model.model.state_dict().items():
        if not torch.equal(value, expected[name]):
            raise ValueError(f"SPD checkpoint contains a different target: {name}")
    return model.eval()


@torch.no_grad()
def evaluate(target, pieces, root, label, author_model=None):
    reference, candidates = comparison.contribution_profiles(target, pieces)
    matched, selected = comparison.select_like_paper(candidates)
    result = {"figure10": comparison.metrics(reference, matched),
              "per_layer": {str(i): {**comparison.metrics(reference[i], matched[i]),
                                      "distinct_selected_pieces": int(selected[i].unique().numel())}
                            for i in range(len(reference))},
              "matrix_relative_errors": {name: core.reconstruction_error(w, p)
                                         for (name, w), p in zip(toy.matrices(target), pieces, strict=True)}}
    comparison.draw(reference.cpu().numpy(), matched.cpu().numpy(),
                    f"{label}\n100 features, 2 layers, frozen 10k-step target",
                    result["figure10"], root / "figure10")
    atomic_save({"target": reference.cpu(), "matched": matched.cpu(), "selected": selected.cpu()},
                root / "figure10_profiles.pt")
    gen = torch.Generator(device="cuda").manual_seed(71001)
    x = sample(4096, 100, gen, "cuda")
    full = target(x)
    meaningful = (full - x @ target.W_E @ target.W_U).square().mean(-1) > 1e-4
    def reconstruction(y):
        error = (y-full).square().mean(-1)
        return {"mse_mean": float(error.mean()), "meaningful_events": int(meaningful.sum()),
                "pass_meaningful": float((error[meaningful] <= 1e-4).float().mean()),
                "max_output_error": float((y-full).abs().max())}
    keep = [torch.ones(len(x), p.count, device="cuda") for p in pieces]
    result["all_kept"] = reconstruction(toy.forward_with_pieces(target, x, pieces, keep)[0])
    if author_model is not None:
        components = {k.replace("-", "."): v for k, v in author_model.components.items()}
        gates = {k.replace("-", "."): v for k, v in author_model.gates.items()}
        a_ref = compute_target_weight_neuron_contributions(author_model.model, 100)
        a_candidates = compute_spd_weight_neuron_contributions(components, author_model.model, 100)
        torch.testing.assert_close(reference, a_ref, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(candidates, a_candidates, atol=1e-6, rtol=1e-5)
        fig = plot_neuron_contribution_pairs(components, author_model.model, 100)
        fig.savefig(root / "authors_figure10.png", dpi=200, bbox_inches="tight")
        fig.savefig(root / "authors_figure10.svg", bbox_inches="tight")
        plt.close(fig)
        _, cache = author_model.forward_with_pre_forward_cache_hooks(x, module_names=list(components))
        ci, _ = calc_causal_importances(cache, {k: c.A for k, c in components.items()}, gates)
        result["native_continuous_ci"] = reconstruction(author_model.forward_with_components(x, components=components, masks=ci))
        result["clipped_ci"] = reconstruction(author_model.forward_with_components(x, components=components,
                                                                                  masks={k: v.clamp(0, 1) for k, v in ci.items()}))
        result["ci_ever_above_0.1"] = {k: int((v > .1).any(0).sum()) for k, v in ci.items()}
        result["ci_mean_count_above_0.1"] = {k: float((v > .1).sum(-1).float().mean()) for k, v in ci.items()}
        result["authors_formula_parity"] = True
    atomic_json(result, root / "evaluation.json")
    print(json.dumps({"evaluation": label, **result}), flush=True)
    return result


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("released", "spd", "method"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--components", type=int, default=400)
    parser.add_argument("--fit-events", type=int, default=16384)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--cache-backend", choices=("dense", "structured"), default="structured")
    parser.add_argument("--fit-layout", choices=("sequential", "joint"), default="joint")
    args = parser.parse_args()
    tracking.start_cli(args)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    assert fingerprint(args.target) == TARGET_SHA
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    set_seed(0)
    raw = yaml.safe_load((Path(args.assets) / "final_config.yaml").read_text())
    cfg = Config(**raw)
    result = {"mode": args.mode, "steps_per_matrix": args.steps, "batch_per_matrix_update": args.batch,
              "components_per_matrix": args.components, "matrices": 4, "target_sha256": TARGET_SHA,
              "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "source_commit": "c6314c9f702b81af593927025aa0ae5aaed4ca4c",
              "precision": "float32; TF32 disabled; no autocast or compilation", "phases": {}}
    if args.mode == "released":
        target = author_target(args.target)
        model = load_author_model(target, cfg, Path(args.assets) / "model_50000.pth")
        pieces = [core.FreePieces(model.components[f"layers-{i}-{kind}"].A,
                                 model.components[f"layers-{i}-{kind}"].B.T)
                  for i in range(2) for kind in ("mlp_in", "mlp_out")]
        result["steps_per_matrix"], result["components_per_matrix"] = cfg.steps, cfg.C
        result["evaluation"] = evaluate(target, pieces, root, "Released SPD - 50,000 steps", model)
    elif args.mode == "spd":
        target = author_target(args.target)
        changes = {"C": args.components, "batch_size": args.batch, "steps": args.steps,
                   "wandb_project": None, "image_freq": None, "image_on_first_step": False,
                   "save_freq": None}
        cfg = Config(**(cfg.model_dump() | changes))
        atomic_json(cfg.model_dump(mode="json"), root / "executed_config.json")
        dataset = ResidualMLPDataset(n_features=100, feature_probability=.01, device="cuda", calc_labels=False)
        loader = DatasetGeneratedDataLoader(dataset, batch_size=args.batch, shuffle=False)
        upstream.tqdm = QuietProgress
        with Meter(root, "fit") as meter:
            upstream.optimize(target, cfg, "cuda", loader, loader, n_eval_steps=100,
                              out_dir=root, plot_results_fn=None)
        result["phases"]["fit"] = meter.result
        result["optimizer_calls"] = args.steps
        result["matrix_updates"] = args.steps * 4
        result["loop_note"] = "Unmodified upstream optimize; W&B/images disabled; print diagnostics and final checkpoint retained."
        if args.evaluate:
            model = load_author_model(target, cfg, root / f"model_{args.steps}.pth")
            pieces = [core.FreePieces(model.components[f"layers-{i}-{kind}"].A,
                                     model.components[f"layers-{i}-{kind}"].B.T)
                      for i in range(2) for kind in ("mlp_in", "mlp_out")]
            result["evaluation"] = evaluate(target, pieces, root, f"SPD retrain - {args.steps:,} steps", model)
    else:
        target = load_target(args.target, "cuda")
        bcfg = objectives.TrainConfig(steps=args.steps, batch=args.batch, log_interval=max(1, args.steps//10))
        atomic_json(asdict(bcfg), root / "executed_config.json")
        with Meter(root, "cache_and_initialization") as prep:
            gen = torch.Generator(device="cuda").manual_seed(101)
            build_cache = objectives.structured.cache_events if args.cache_backend == "structured" else toy.cache_events
            cache = build_cache(target, sample(args.fit_events, 100, gen, "cuda"))
            result["cache_backend"] = args.cache_backend
            result["cache_bytes"] = objectives.structured.cache_bytes(cache)
            initial, init_info = [], {}
            for name, weight in toy.matrices(target):
                p, info = objectives.prepare_initial(weight, args.components, bcfg)
                initial.append(p)
                init_info[name] = info
        result["phases"]["cache_and_initialization"] = prep.result
        result["initialization"] = init_info
        result["optimizer"] = "projected_adam"
        pieces, histories = [], {}
        with Meter(root, "fit") as meter:
            named_weights = toy.matrices(target)
            if args.fit_layout == "joint":
                pieces, rows = objectives.fit(
                    [cache["matrices"][name] for name, _ in named_weights],
                    [weight for _, weight in named_weights], initial, bcfg,
                    log=lambda row: print(json.dumps({"matrix": named_weights[row["matrix_index"]][0], **row}), flush=True))
                histories = {name: rows[i] for i, (name, _) in enumerate(named_weights)}
            else:
                for i, (name, weight) in enumerate(named_weights):
                    p, history = objectives.fit_matrix(cache["matrices"][name], weight, initial[i], bcfg,
                                                          log=lambda row, name=name: print(json.dumps({"matrix": name, **row}), flush=True))
                    pieces.append(p)
                    histories[name] = history
            atomic_json(histories, root / "training.json")
            atomic_save({"pieces": [{"read": p.read.cpu(), "write": p.write.cpu()} for p in pieces]}, root / "pieces.pt")
        result["phases"]["fit"] = meter.result
        result["fit_layout"] = args.fit_layout
        result["optimizer_calls"] = args.steps * (1 if args.fit_layout == "joint" else 4)
        result["matrix_updates"] = args.steps * 4
        result["fit_events"] = args.fit_events
        if args.evaluate:
            result["evaluation"] = evaluate(target, pieces, root, f"Method ({args.cache_backend}) - {args.steps:,} steps per matrix")
    if "fit" in result["phases"]:
        result["seconds_per_four_matrix_update"] = result["phases"]["fit"]["wall_seconds"] / args.steps
        result["fit_plus_setup_seconds"] = sum(v["wall_seconds"] for v in result["phases"].values())
    atomic_json(result, root / "result.json")
    print(json.dumps({"benchmark_complete": args.mode, **{k: v for k, v in result.items() if k != "evaluation"}}), flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
