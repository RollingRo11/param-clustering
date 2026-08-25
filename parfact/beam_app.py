"""Run parfact decompositions on Beam GPUs.

Run from this directory (the working dir is synced to the container):

    python beam_app.py                          # variant B, norm layer
    python beam_app.py --sweep                  # one container per config
    python beam_app.py --train_steps 3000       # quick smoke run

Artifacts stay on a Beam DurableDisk; nothing large is downloaded. Use
`python beam_app.py --ls` to see what's on the disk.

NOTE: Beam Volumes are deliberately NOT used here. Verified in this
workspace: a GPU container's Volume mount silently no-ops -- writes vanish
and CPU-written files are invisible -- so anything a GPU job writes to a
Volume is lost. DurableDisk mounts for real (it leaves a
`.beta9-durable-disk` marker) and persists across GPU containers.
"""
import argparse
import base64
import gzip
import subprocess
import sys
from pathlib import Path

from beam import Image, function
from beta9.type import DurableDisk

image = (
    Image(python_version="python3.12")
    .add_python_packages(["torch", "matplotlib", "numpy"])
)

# Persistent block storage, mounted at /data in every container below.
# `size` is honoured only when the disk is first created -- changing it here
# is silently ignored (this one is still the 10Gi it was created with; check
# with `beam disk list`). Size a new disk correctly up front.
DISK = [DurableDisk(name="parfact-disk", size="10Gi", mount_path="/data")]

# The return value travels through Beam's task queue, which cannot carry
# artifacts -- ~20MB of them fails the task with DEADLINE_EXCEEDED. Only
# small JSON comes back this way; tensors stay on the disk.
MAX_RETURN_BYTES = 1024 * 1024


@function(
    name="parfact",
    image=image,
    gpu="A10G",
    cpu=4,
    memory="16Gi",
    timeout=-1,          # decompositions run long; -1 disables the task timeout
    retries=0,           # a retrying zombie holds the GPU quota; fail loudly
    disks=DISK,
)
def run(variant: str = "B", norm: str = "layer", k_factors: int = 8,
        c_groups: int = 6, train_steps: int = 100_000):
    """One run_parfact.py invocation; returns its artifacts as bytes."""
    tag = f"{variant}_{norm}_K{k_factors}_C{c_groups}"
    out = Path("/data/runs") / tag
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "run_parfact.py",
        "--variant", variant,
        "--norm", norm,
        "--k_factors", str(k_factors),
        "--c_groups", str(c_groups),
        "--train_steps", str(train_steps),
        "--ckpt", "/data/induction_model.pt",   # trained once, reused
        "--out", str(out),
        "--device", "cuda",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Artifacts stay on /data. Only small files ride the return channel.
    artifacts, left_on_disk = {}, []
    for f in sorted(out.iterdir()):
        if not f.is_file():
            continue
        size = f.stat().st_size
        if size > MAX_RETURN_BYTES:
            left_on_disk.append((f.name, size))
            continue
        artifacts[f.name] = base64.b64encode(gzip.compress(f.read_bytes())).decode()
    return {"tag": tag, "path": str(out),
            "artifacts": artifacts, "on_disk": left_on_disk}


@function(image=Image(python_version="python3.12"), cpu=1, memory="2Gi",
          disks=DISK, timeout=300, retries=0)
def ls():
    """List the disk from a CPU container -- no GPU needed to browse results."""
    import os
    return sorted(
        (os.path.join(d, f), os.path.getsize(os.path.join(d, f)))
        for d, _, fs in os.walk("/data") for f in fs)


def save(res, root=Path("out")):
    """Write the small returned artifacts under ./out/<tag>/; report the rest."""
    d = root / res["tag"]
    d.mkdir(parents=True, exist_ok=True)
    for name, blob in res["artifacts"].items():
        (d / name).write_bytes(gzip.decompress(base64.b64decode(blob)))
    print(f"wrote {len(res['artifacts'])} files to {d}")
    for name, size in res["on_disk"]:
        print(f"  left on disk: {res['path']}/{name} ({size/1e6:.1f} MB)")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls", action="store_true",
                    help="list artifacts on the durable disk and exit")
    ap.add_argument("--sweep", action="store_true",
                    help="fan out over variants x norms in parallel containers")
    ap.add_argument("--variant", default="B")
    ap.add_argument("--norm", default="layer")
    ap.add_argument("--train_steps", type=int, default=100_000)
    args = ap.parse_args()

    if args.ls:
        for path, size in ls.remote():
            print(f"{size:>14,}  {path}")
        sys.exit(0)

    if args.sweep:
        grid = [(v, n) for v in ("A", "B", "C") for n in ("layer", "fisher")]
        # Deliberately serial, NOT .map(). Verified: when several containers
        # mount the same DurableDisk at once it splits into divergent
        # replicas -- 4 concurrent writers, and one's file was silently
        # discarded on reconcile. Parallelise only with one disk per config
        # or an S3-backed CloudBucket.
        for v, n in grid:
            save(run.remote(variant=v, norm=n, train_steps=args.train_steps))
    else:
        save(run.remote(variant=args.variant, norm=args.norm,
                        train_steps=args.train_steps))
