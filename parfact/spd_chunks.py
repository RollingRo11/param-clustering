"""Drive the C=600 SPD run as short resumable chunks on the reserved H100.

Every prior strategy lost the whole run to a single failure: .remote() was
cancelled when its gRPC stream broke (8 min), a sandbox expired on its TTL,
and a nohup'd process died when its container was recycled. Short calls are
the one thing that has completed reliably, so each chunk is ~4 min and all
state lives on the durable disk.

    python spd_chunks.py verify    # 2x2000 == 1x4000 ?
    python spd_chunks.py run       # the real 100k-step Table 4 run
"""
import subprocess, sys, time
from beam import Image, function
from beta9.type import DurableDisk

# torch pinned: today's marketplace L40S machines run an older NVIDIA
# driver (CUDA 12.8); latest torch wheels need newer. 2.7.1 ships cu126.
IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))
GPU = dict(image=IMG, gpu="L40S", pool="ondemand-l40s", cpu=8,
           memory="32Gi", timeout=1800, retries=0)


def _chunk_impl(out, steps, chunk_steps, fresh, seed, extra,
                script="spd_toy_resumable.py"):
    import shutil, os, subprocess as sp, sys as s
    name = sp.check_output(["nvidia-smi", "--query-gpu=name",
                            "--format=csv,noheader"]).decode().strip()
    assert "L40S" in name, f"expected L40S, got {name}"   # fail fast
    if fresh:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    p = sp.run([s.executable, "-u", script,
                "--c_per_module", "100", "--steps", str(steps),
                "--chunk", str(chunk_steps), "--ckpt", "induction_model_100k.pt",
                "--out", out, "--world", "1", "--seed", str(seed)] + list(extra),
               capture_output=True, text=True)
    tail = (p.stdout or "").strip().splitlines()[-6:]
    done = os.path.exists(f"{out}/spd_state.pt")
    return {"gpu": name, "rc": p.returncode, "secs": round(time.time() - t0, 1),
            "tail": tail, "done": done, "err": (p.stderr or "")[-400:]}


# One disk PER concurrent run: two containers mounting one DurableDisk at the
# same time split into divergent replicas and silently drop a writer's data.
@function(name="spd-chunk-paper", disks=[DurableDisk(
    name="spd-c600-paper", size="10Gi", mount_path="/data")], **GPU)
def chunk_paper(out: str, steps: int, chunk_steps: int, fresh: bool = False,
                seed: int = 0, extra: list = []):
    return _chunk_impl(out, steps, chunk_steps, fresh, seed, extra)


@function(name="spd-chunk-s1", disks=[DurableDisk(
    name="spd-c600-s1", size="10Gi", mount_path="/data")], **GPU)
def chunk_s1(out: str, steps: int, chunk_steps: int, fresh: bool = False,
             seed: int = 0, extra: list = []):
    return _chunk_impl(out, steps, chunk_steps, fresh, seed, extra)


@function(name="spd-chunk-concat", disks=[DurableDisk(
    name="spd-c600-concat", size="10Gi", mount_path="/data")], **GPU)
def chunk_concat(out: str, steps: int, chunk_steps: int, fresh: bool = False,
                 seed: int = 0, extra: list = []):
    return _chunk_impl(out, steps, chunk_steps, fresh, seed, extra,
                       script="spd_toy_concat.py")


CHUNKS = {"paper": chunk_paper, "s1": chunk_s1, "concat": chunk_concat}


def verify():
    a = chunk.remote("/data/runs/vA", 4000, 4000, True)          # single run
    print("SINGLE:", *a["tail"][-2:], sep="\n  ")
    chunk.remote("/data/runs/vB", 4000, 2000, True)              # 2 x 2000
    b = chunk.remote("/data/runs/vB", 4000, 2000, False)
    print("CHUNKED:", *b["tail"][-2:], sep="\n  ")


def run():
    import sys as _s
    which = _s.argv[2]                     # "paper" or "s1"
    seed = int(_s.argv[3]) if len(_s.argv) > 3 else 0
    # NEVER wipe by default: chunks resume from <out>/resume.pt if present,
    # which on an empty dir is identical to a fresh start. A restart-after-
    # crash driver once wiped 96k steps of checkpoints via fresh=True.
    extra = [a for a in _s.argv[4:] if a != "fresh"]
    fn = CHUNKS[which]
    out, steps, cs = f"/data/runs/spd_C600_{which}", 100_000, 4000
    for i in range(steps // cs):
        # Gateway blips kill .remote() (raise, or return None); chunks are
        # idempotent thanks to resume.pt, so retry the same chunk.
        r = None
        for attempt in range(5):
            try:
                r = fn.remote(out, steps, cs, fresh=False, seed=seed,
                              extra=extra)
            except Exception as e:
                print(f"chunk {i} attempt {attempt}: {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
            if r is not None:
                break
            time.sleep(60)
        if r is None or r["rc"] != 0:
            print(f"chunk {i} FAILED:", r and r["err"], flush=True)
            break
        print(f"[{i+1}/{steps//cs}] {r['secs']}s  {r['tail'][-1] if r['tail'] else ''}",
              flush=True)
        if r["done"]:
            print("FINISHED:", *r["tail"], sep="\n  ", flush=True)
            break


if __name__ == "__main__":
    {"verify": verify, "run": run}[sys.argv[1]]()
