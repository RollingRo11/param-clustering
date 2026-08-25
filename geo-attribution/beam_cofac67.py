"""Co-factorization on VPD's 67M 4-layer Pile target, with VPD comparison.

Stages (each resumable, artifacts on durable disk cofac67):
    python beam_cofac67.py fetch     # target ckpt + VPD decomp from wandb
    python beam_cofac67.py sanity    # model loads, 24 modules, finite logits
    (collect / fit / eval / compare added as they are verified)
"""
import json
import sys
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "numpy", "wandb",
                             "transformers", "datasets", "einops", "zstandard"]))
DISK = [DurableDisk(name="cofac67", size="50Gi", mount_path="/data")]
GPU = dict(image=IMG, gpu="L40S", pool="ondemand-l40s", cpu=8,
           memory="48Gi", disks=DISK, timeout=3600, retries=0)


@function(name="cofac67-fetch", image=IMG, cpu=4, memory="16Gi",
          disks=DISK, timeout=3600, retries=0)
def fetch():
    import os, urllib.request
    out = {}
    for run_id, fname, dst in (
            ("goodfire/spd/t-9d2b8f02", "model_step_99999.pt",
             "/data/target"),
            ("goodfire/spd/s-55ea3f9b", "model_400000.pth",
             "/data/vpd")):
        os.makedirs(dst, exist_ok=True)
        path = f"{dst}/{fname}"
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            out[run_id] = f"cached {os.path.getsize(path)}"
            continue
        url = f"https://api.wandb.ai/files/{run_id}/{fname}"
        urllib.request.urlretrieve(url, path)
        out[run_id] = f"downloaded {os.path.getsize(path)}"
    return out


@function(name="cofac67-sanity", **GPU)
def sanity():
    import subprocess as sp, torch, os
    import sensor_study67 as S67
    S67.CKPT = __import__("pathlib").Path("/data/target/model_step_99999.pt")
    model, sd = None, None
    dev = "cuda"
    model = S67.load67(dev, "plain")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids = tok("The capital of France is", return_tensors="pt")["input_ids"].to(dev)
    with torch.no_grad():
        logits = model(ids)
    top = tok.decode(logits[0, -1].argmax())
    vpd_sd = torch.load("/data/vpd/model_400000.pth", map_location="cpu",
                        weights_only=False)
    n_vpd_keys = len(vpd_sd) if hasattr(vpd_sd, "__len__") else -1
    return {"modules": S67.MODULES, "n_modules": len(S67.MODULES),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "next_token": top, "vpd_keys": n_vpd_keys,
            "vpd_key_sample": sorted(list(vpd_sd))[:6] if n_vpd_keys > 0 else []}


@function(name="cofac67-atoms", **GPU)
def atoms():
    import cofac67
    return cofac67.atoms_prep()


@function(name="cofac67-verify", **GPU)
def verify():
    import cofac67
    return cofac67.verify()


@function(name="cofac67-collect", **GPU)
def collect(chunk_id: int, n_chunks: int):
    import cofac67
    return cofac67.collect_chunk(chunk_id, n_chunks)


@function(name="cofac67-fit", **GPU)
def fit():
    import cofac67
    return cofac67.fit()


@function(name="cofac67-eval", **GPU)
def evalkl():
    import cofac67
    return cofac67.eval_klkeep()


@function(name="cofac67-oracle", **GPU)
def oracle():
    import cofac67
    return cofac67.eval_oracle()


GPU4090 = dict(image=IMG, gpu="RTX4090", cpu=8, memory="24Gi",
               disks=DISK, timeout=3600, retries=0)


@function(name="cofac67-bench", **GPU4090)
def bench4090():
    import cofac67
    return cofac67.bench_collect()


@function(name="cofac67-bench-l40s", **GPU)
def benchl40s():
    import cofac67
    return cofac67.bench_collect()


GPU5090 = dict(image=IMG, gpu="RTX5090", cpu=8, memory="24Gi",
               disks=DISK, timeout=3600, retries=0)


@function(name="cofac67-bench-5090", **GPU5090)
def bench5090():
    import cofac67
    return cofac67.bench_collect()


@function(name="cofac67-prestage-shard", image=IMG, cpu=8, memory="32Gi",
          disks=DISK, timeout=3600 * 3, retries=0)
def prestage_shard(shard_id: int):
    import cofac67
    return cofac67.prestage_shard(shard_id)


@function(name="cofac67-collect-big", **GPU4090)
def collect_big(chunk_id: int, n_chunks: int):
    import cofac67
    return cofac67.collect_chunk_big(chunk_id, n_chunks)


@function(name="cofac67-collect-span", **GPU)
def collect_span(start_file: int, end_file: int):
    import cofac67
    return cofac67.collect_span(start_file, end_file)


@function(name="cofac67-interp", image=IMG, cpu=8, memory="32Gi",
          disks=DISK, timeout=3600, retries=0)
def interp():
    import cofac67
    return cofac67.interp_report()


@function(name="cofac67-vpdkl", **GPU)
def vpdkl():
    import cofac67
    return cofac67.eval_vpd_klkeep()


@function(name="cofac67-compare", **GPU)
def compare():
    import cofac67
    return cofac67.compare_vpd()


if __name__ == "__main__":
    import time
    cmd = sys.argv[1]
    if cmd == "collect_span":
        # args: total_files span_size worker_idx n_workers
        total_files = int(sys.argv[2])
        span = int(sys.argv[3])
        widx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        nw = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        spans = [(a, min(a + span, total_files))
                 for a in range(0, total_files, span)]
        for si, (a, b) in enumerate(spans):
            if si % nw != widx:
                continue
            r = None
            for attempt in range(4):
                try:
                    r = collect_span.remote(a, b)
                except Exception as e:
                    print(f"span {a}-{b} attempt {attempt}: "
                          f"{str(e)[:100]}", flush=True)
                if r is not None:
                    break
                time.sleep(60)
            print(f"[span {a}-{b}] {json.dumps(r)}", flush=True)
            if r is None:
                break
        sys.exit(0)
    if cmd == "collect_big":
        n_chunks = int(sys.argv[2])
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        for i in range(start, n_chunks, step):
            r = None
            for attempt in range(4):
                try:
                    r = collect_big.remote(i, n_chunks)
                except Exception as e:
                    print(f"chunk {i} attempt {attempt}: "
                          f"{str(e)[:100]}", flush=True)
                if r is not None:
                    break
                time.sleep(60)
            print(f"[{i}] {json.dumps(r)}", flush=True)
            if r is None:
                break
        sys.exit(0)
    if cmd == "collect":
        n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 32
        for i in range(n_chunks):
            r = None
            for attempt in range(4):
                try:
                    r = collect.remote(i, n_chunks)
                except Exception as e:
                    print(f"chunk {i} attempt {attempt}: {e}", flush=True)
                if r is not None:
                    break
                time.sleep(60)
            print(f"[{i+1}/{n_chunks}]", json.dumps(r), flush=True)
            if r is None:
                break
        sys.exit(0)
    fn = {"fetch": fetch, "sanity": sanity, "atoms": atoms,
          "verify": verify, "fit": fit, "eval": evalkl,
          "compare": compare, "oracle": oracle,
          "vpdkl": vpdkl, "interp": interp,
          "bench4090": bench4090, "benchl40s": benchl40s,
          "bench5090": bench5090,
          }[cmd] if cmd != "prestage_shard" else None
    if cmd == "prestage_shard":
        r = prestage_shard.remote(int(sys.argv[2]))
        print(json.dumps(r, indent=1) if r else "FAILED")
        sys.exit(0 if r else 1)
    r = fn.remote()
    print(json.dumps(r, indent=1, default=str) if r else "FAILED")
