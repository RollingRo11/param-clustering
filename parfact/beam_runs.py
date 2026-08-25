"""SPD vs parfact co-factorization at C=600 on the toy induction model, on Beam.

Three phases, all writing to the durable disk at /data:

    python beam_runs.py target    # train the shared 6-matrix induction model
    python beam_runs.py spd       # Christensen & Riggs Smith SPD, C=600
    python beam_runs.py cofact    # parfact co-factorization, K=1200 C=600
    python beam_runs.py ls        # what's on the disk (CPU container)

Both decompositions must target the SAME checkpoint, so `target` runs first
and the other two read /data/induction_model.pt.

A10G is deliberate: d_model=16 makes this kernel-launch bound, not compute
bound, so a larger GPU costs more per hour and finishes no sooner. It is
also Beam's only serverless GPU -- the rest bill hourly on-demand.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))

# The shared target model rides along with the normal working-dir file sync
# (31KB), so the two decompositions need NOTHING from each other and can run
# at the same time. They must not share a disk: several containers mounting
# one DurableDisk concurrently split into divergent replicas and one side's
# writes are silently dropped (measured -- see beam_app.py).
CKPT = "induction_model_100k.pt"
DISK = [DurableDisk(name="parfact-disk", size="10Gi", mount_path="/data")]
SPD_DISK = [DurableDisk(name="spd-c600", size="10Gi", mount_path="/data")]
COF_DISK = [DurableDisk(name="cofact-c600", size="10Gi", mount_path="/data")]

GPU = dict(image=IMG, gpu="A10G", cpu=4, memory="16Gi",
           timeout=-1, retries=0)


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@function(name="parfact-target", disks=DISK, **GPU)
def target(steps: int = 100_000):
    """Train the shared induction model at the paper's 100k steps."""
    import os, shutil
    for junk in ("/data/probe", "/data/concurrent"):   # earlier storage tests
        shutil.rmtree(junk, ignore_errors=True)
    sh([sys.executable, "induction_model.py", "--steps", str(steps),
        "--out", "/data/induction_model.pt", "--device", "cuda"])
    return {"ckpt": "/data/induction_model.pt",
            "bytes": os.path.getsize("/data/induction_model.pt")}


@function(name="parfact-spd", disks=SPD_DISK, **GPU)
def spd(c_per_module: int = 100, steps: int = 100_000):
    """SPD exactly as Christensen & Riggs Smith Table 4: C=100 x 6 = 600."""
    out = Path("/data/runs/spd_C600")
    out.mkdir(parents=True, exist_ok=True)
    # --world 1: one A10G, so the global batch of 1024 stays unsplit and
    # matches the paper's setting rather than being halved across ranks.
    sh([sys.executable, "spd_toy.py",
        "--c_per_module", str(c_per_module), "--steps", str(steps),
        "--ckpt", CKPT, "--out", str(out), "--world", "1"])
    return {"out": str(out),
            "files": sorted(f.name for f in out.iterdir() if f.is_file())}


@function(name="parfact-cofact", disks=COF_DISK, **GPU)
def cofact(k_factors: int = 1200, c_groups: int = 600):
    """Co-factorization at the pairing compare_components.py expects."""
    out = Path(f"/data/runs/B_layer_K{k_factors}_C{c_groups}")
    out.mkdir(parents=True, exist_ok=True)
    sh([sys.executable, "run_parfact.py",
        "--variant", "B", "--norm", "layer",
        "--k_factors", str(k_factors), "--c_groups", str(c_groups),
        "--ckpt", CKPT, "--out", str(out), "--device", "cuda"])
    metrics = (out / "metrics.json").read_text()
    print(metrics, flush=True)
    return {"out": str(out), "metrics": metrics}


def _lister(disk):
    @function(image=Image(python_version="python3.12"), cpu=1, memory="2Gi",
              disks=disk, timeout=300, retries=0)
    def _ls():
        import os
        return sorted((os.path.join(d, f), os.path.getsize(os.path.join(d, f)))
                      for d, _, fs in os.walk("/data") for f in fs)
    return _ls


ls_target, ls_spd, ls_cofact = _lister(DISK), _lister(SPD_DISK), _lister(COF_DISK)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase",
                    choices=["target", "spd", "cofact", "ls"])
    ap.add_argument("--disk", choices=["target", "spd", "cofact"],
                    default="spd", help="which disk `ls` should list")
    ap.add_argument("--steps", type=int, default=100_000)
    args = ap.parse_args()

    if args.phase == "ls":
        lister = {"target": ls_target, "spd": ls_spd,
                  "cofact": ls_cofact}[args.disk]
        for path, size in lister.remote():
            print(f"{size:>14,}  {path}")
    elif args.phase == "target":
        print(target.remote(steps=args.steps))
    elif args.phase == "spd":
        print(spd.remote(steps=args.steps))
    else:
        print(cofact.remote())
