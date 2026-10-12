"""Assign each raw CUDA activity once, through its CPU launch correlation."""
import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

from parauto_io import method_directory


def trace_breakdown(path):
    events = json.loads(path.read_text())["traceEvents"]
    stages = sorted((e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                     and e["name"].startswith("stage/")), key=lambda e: e["ts"])
    starts = [e["ts"] for e in stages]
    for left, right in zip(stages, stages[1:]):
        assert left["ts"] + left["dur"] <= right["ts"] + .01, "Overlapping stage annotations"
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    cpu_ops = {e["args"]["External id"]: e for e in events
               if e.get("ph") == "X" and e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    gpu_us, cpu_us, ops_us, counts = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(int)
    for stage in stages:
        cpu_us[stage["name"]] += stage["dur"]
        counts[stage["name"]] += 1
    count = counts["stage/backward"]
    assert count == 20, count
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        args = event.get("args", {})
        launch = launches.get(args.get("correlation"))
        op = cpu_ops.get(args.get("External id"))
        label = "other"
        if launch is not None:
            i = bisect.bisect_right(starts, launch["ts"]) - 1
            if i >= 0 and launch["ts"] <= stages[i]["ts"] + stages[i]["dur"]:
                label = stages[i]["name"]
        gpu_us[label] += event["dur"]
        ops_us[op["name"] if op is not None else event["name"]] += event["dur"]
    assert sum(gpu_us.values()) > 0
    return {"updates": count, "gpu_ms_per_update": sum(gpu_us.values()) / count / 1000,
            "stage_gpu_ms_per_update": {k: v/count/1000 for k, v in gpu_us.items()},
            "stage_cpu_elapsed_ms_per_update": {k: v/count/1000 for k, v in cpu_us.items()},
            "top_launch_operators": sorted(({"name": k, "gpu_ms_per_update": v/count/1000}
                                              for k, v in ops_us.items()), key=lambda x: -x["gpu_ms_per_update"])[:15]}


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    tracking.start_cli(args)
    result = {}
    for mode in ("spd", "method"):
        directory = method_directory(args.root) if mode == "method" else args.root / mode
        run = json.loads((directory / "profile.json").read_text())
        traces = {p.name: trace_breakdown(p) for p in sorted(directory.glob("*.trace.json"))}
        combined = defaultdict(float)
        for trace in traces.values():
            for name, duration in trace["stage_gpu_ms_per_update"].items():
                combined[name] += duration
        result[mode] = {"warm_ms_per_four_matrix_update": run["warm_ms_per_four_matrix_update"],
                        "projected_10k_fit_seconds": run["projected_10k_fit_seconds"],
                        "warm_timings": {k: v for k, v in run["phases"].items() if k.endswith("_warm")},
                        "gpu_ms_per_four_matrix_update_under_profiler": sum(combined.values()),
                        "stage_gpu_ms_per_four_matrix_update": dict(combined), "traces": traces}
    result["method_time_over_SPD"] = result["method"]["warm_ms_per_four_matrix_update"] / result["spd"]["warm_ms_per_four_matrix_update"]
    (args.root / "breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# SPD/method timing profile", "", "400 components per matrix, batch 2,048; both methods measured sequentially on one separate V100.", "",
             "| Warm throughput | SPD | method |", "|---|---:|---:|",
             f"| Milliseconds per update of all four matrices | {result['spd']['warm_ms_per_four_matrix_update']:.3f} | {result['method']['warm_ms_per_four_matrix_update']:.3f} |",
             f"| Projected 10k fit time (seconds) | {result['spd']['projected_10k_fit_seconds']:.2f} | {result['method']['projected_10k_fit_seconds']:.2f} |", "",
             "The 10k figures are extrapolations from 800 timed updates after 200 warm-up updates, not additional completed 10k training runs. "
             "The profiler was disabled during throughput measurement. The separate traces cover 20 early-training updates after warm-up; "
             "they do not load intermediate 10k weights. method retains the 50k p/LR schedules. The production 50k benchmark was not modified.", "",
             "## GPU activity breakdown", "", "Each CUDA kernel/memcpy/memset is counted once and assigned to a stage through its CPU launch correlation ID. "
             "GPU annotation ranges and inclusive operator summaries are excluded to avoid double counting. "
             "Autograd worker-thread launches are assigned by launch time inside the backward stage. "
             "These are device activity durations under instrumentation, not uninstrumented wall-clock fractions.", ""]
    for mode in ("spd", "method"):
        lines.extend([f"### {mode}", "", "| Stage | GPU ms / four-matrix update | Share of recorded GPU activity |",
                      "|---|---:|---:|"])
        total = result[mode]["gpu_ms_per_four_matrix_update_under_profiler"]
        for name, duration in sorted(result[mode]["stage_gpu_ms_per_four_matrix_update"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name.removeprefix('stage/')} | {duration:.3f} | {100*duration/total:.1f}% |")
    lines.extend(["", "Complete per-matrix throughput, CPU stage elapsed times, raw CUDA traces and launch-operator attribution are retained beside this report.", ""])
    (args.root / "profile_report.md").write_text("\n".join(lines))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "traces"} if isinstance(v, dict) else v
                      for k, v in result.items()}), flush=True)


if __name__ == "__main__":
    tracking.run_main(main)
