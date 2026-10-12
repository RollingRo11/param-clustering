"""Warm throughput and short CPU/CUDA traces without modifying the 50k runs."""
from __future__ import annotations

import argparse
import builtins
import functools
import json
import time
from pathlib import Path

import benchmark_spd as bench
import torch


class Instrument:
    def __init__(self, root):
        self.root = root
        self.active = False
        self.tracing = False
        self.depth = 0
        self.profiler = None
        self.count = 0
        self.label = ""
        self.results = {}
        self.originals = []

    def wrap(self, owner, name, label, boundary=False):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if self.tracing and not self.depth:
                self.depth += 1
                try:
                    with torch.profiler.record_function(label):
                        result = original(*args, **kwargs)
                finally:
                    self.depth -= 1
            else:
                result = original(*args, **kwargs)
            if boundary and self.active:
                self.count += 1
                if self.profiler is not None:
                    self.profiler.step()
                elif self.count == 200:
                    torch.cuda.synchronize()
                    self.start = time.perf_counter()
                elif self.count == 1000:
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - self.start
                    self.results[self.label] = {"measured_updates": 800, "warmup_updates": 200,
                                                "wall_seconds": elapsed, "ms_per_update": elapsed / .8}
            return result

        setattr(owner, name, wrapped)

    def capture(self, prof):
        rows = []
        for event in prof.key_averages():
            # CUDA kernel events and GPU annotation spans also appear here.
            # Mixing them with CPU operator aggregates double-counts activity.
            # Raw trace correlation in analyze_spd_profile.py is authoritative.
            if not str(event.device_type).endswith(".CPU"):
                continue
            rows.append({"name": event.key, "calls": event.count,
                         "cpu_total_us": event.cpu_time_total, "cpu_self_us": event.self_cpu_time_total,
                         "gpu_total_us": event.device_time_total, "gpu_self_us": event.self_device_time_total})
        self.results[self.label] = {"profiled_updates": 20,
            "cpu_stages": [{k: v for k, v in r.items() if not k.startswith("gpu_")}
                           for r in rows if r["name"].startswith("stage/")],
            "operators_by_gpu_self": sorted(rows, key=lambda r: r["gpu_self_us"], reverse=True)[:30],
            "operators_by_cpu_self": sorted(rows, key=lambda r: r["cpu_self_us"], reverse=True)[:30],
            "gpu_accounting": "Use analyze_spd_profile.py on raw trace; CPU scope totals miss some autograd worker-thread attribution."}
        if not any(r["gpu_self_us"] > 0 for r in rows):
            raise RuntimeError("CUDA profiler did not collect GPU events")
        prof.export_chrome_trace(str(self.root / f"{self.label}.trace.json"))
        (self.root / f"{self.label}.operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35))

    def run(self, label, fn, trace):
        self.label, self.count = label, 0
        self.active, self.tracing = True, trace
        try:
            if trace:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=200, warmup=3, active=20, repeat=1),
                    on_trace_ready=self.capture, record_shapes=False, profile_memory=False,
                ) as prof:
                    self.profiler = prof
                    fn()
            else:
                fn()
        finally:
            self.active = self.tracing = False
            self.profiler = None
        print(json.dumps({"profile_phase": label, "updates_executed": self.count,
                          "result": self.results[label]}), flush=True)


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("spd", "method"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    bench.set_seed(0)
    assert bench.fingerprint(args.target) == bench.TARGET_SHA
    inst = Instrument(args.output)
    inst.wrap(torch.Tensor, "backward", "stage/backward")
    if args.mode == "spd":
        inst.wrap(torch.optim.AdamW, "step", "stage/optimizer", boundary=True)
        inst.wrap(bench.ComponentModel, "forward_with_pre_forward_cache_hooks", "stage/target_forward")
        inst.wrap(bench.upstream, "calc_causal_importances", "stage/importance_gates")
        inst.wrap(bench.upstream, "calculate_losses", "stage/losses_and_masked_forwards")
        target = bench.author_target(args.target)
        raw = bench.yaml.safe_load((Path(args.assets) / "final_config.yaml").read_text())
        dataset = bench.ResidualMLPDataset(n_features=100, feature_probability=.01, device="cuda", calc_labels=False)
        loader = bench.DatasetGeneratedDataLoader(dataset, batch_size=2048, shuffle=False)
        bench.upstream.tqdm = bench.QuietProgress
        for trace in (False, True):
            cfg = bench.Config(**(raw | {"steps": 250 if trace else 1000, "wandb_project": None,
                                        "image_freq": None, "image_on_first_step": False, "save_freq": None}))
            inst.run("spd_trace" if trace else "spd_warm",
                     lambda: bench.upstream.optimize(target, cfg, "cuda", loader, loader,
                                                      n_eval_steps=100, out_dir=None), trace)
    else:
        inst.wrap(torch.optim.Adam, "step", "stage/optimizer", boundary=True)
        inst.wrap(bench.objectives, "project_", "stage/post_step_projection")
        inst.wrap(bench.core, "projected_pieces", "stage/differentiable_projection")
        inst.wrap(bench.objectives, "slice_entry", "stage/cache_batch_gather")
        inst.wrap(bench.objectives, "scores_of", "stage/component_scores")
        inst.wrap(bench.objectives, "concentration", "stage/sparse_penalty")
        target = bench.load_target(args.target, "cuda")
        cfg = bench.objectives.TrainConfig(steps=50000, batch=2048, log_interval=50000)
        gen = torch.Generator(device="cuda").manual_seed(101)
        cache = bench.objectives.structured.cache_events(target, bench.sample(16384, 100, gen, "cuda"))
        initial = [bench.objectives.prepare_initial(w, 400, cfg)[0]
                   for name, w in bench.toy.matrices(target)]
        for trace in (False, True):
            # Shorten observation only; preserve the 50k run's p/LR schedules.
            stop = 250 if trace else 1000
            named = bench.toy.matrices(target)
            inst.run("method_trace" if trace else "method_warm",
                     lambda: bench.objectives.fit(
                         [cache["matrices"][name] for name, _ in named],
                         [weight for _, weight in named], initial, cfg, log=lambda row: None, stop_after=stop), trace)
    result = {"mode": args.mode, "gpu": torch.cuda.get_device_name(), "batch": 2048, "components": 400,
              "torch": torch.__version__, "phases": inst.results,
              "note": "Separate profiling run: early structured training, not saved 10k-step weights. Throughput excludes first 200 updates and profiling overhead. Our method keeps its 50k schedules; SPD uses its constant LR. No full-run files or training code were changed."}
    warm = [v for k, v in inst.results.items() if k.endswith("_warm")]
    result["warm_ms_per_four_matrix_update"] = sum(v["ms_per_update"] for v in warm)
    result["projected_10k_fit_seconds"] = result["warm_ms_per_four_matrix_update"] * 10
    bench.atomic_json(result, args.output / "profile.json")
    print(json.dumps({"profile_complete": args.mode, **{k: v for k, v in result.items() if k != "phases"}}), flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
