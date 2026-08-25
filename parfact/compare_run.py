"""Run compare_components.py against both decompositions on Beam.

The two artifacts live on separate durable disks (they had to -- concurrent
writers to one disk silently diverge), so this mounts both at once. That is
safe: it is a single container reading, not competing writers.

Runs on serverless RTX4090 by default so it does not contend with the
reserved H100 that the SPD chunks are using.

    python compare_run.py <spd_dir_name>     # e.g. spd_C600, or vC to smoke-test
"""
import sys
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))

DISKS = [DurableDisk(name="spd-c600", size="10Gi", mount_path="/spd"),
         DurableDisk(name="cofact-c600", size="10Gi", mount_path="/cofact")]


@function(name="compare", image=IMG, gpu="RTX4090", cpu=4, memory="16Gi",
          disks=DISKS, timeout=1800, retries=0)
def compare(spd_dir: str = "spd_C600",
            cofac_dir: str = "B_layer_K1200_C600"):
    import subprocess, sys as s, os
    spd, cof = f"/spd/runs/{spd_dir}", f"/cofact/runs/{cofac_dir}"
    for p in (f"{spd}/spd_state.pt", f"{cof}/factorization.pt"):
        if not os.path.exists(p):
            return {"missing": p}
    p = subprocess.run(
        [s.executable, "-u", "compare_components.py",
         "--cofac", cof, "--spd", spd,
         "--ckpt", "induction_model_100k.pt", "--device", "cuda:0"],
        capture_output=True, text=True)
    return {"rc": p.returncode, "out": p.stdout, "err": p.stderr[-1500:]}


if __name__ == "__main__":
    r = compare.remote(spd_dir=sys.argv[1] if len(sys.argv) > 1 else "spd_C600")
    if r is None:
        print("FAILED: returned None")
    elif "missing" in r:
        print("MISSING:", r["missing"])
    else:
        print(r["out"])
        if r["rc"]:
            print("--- stderr ---\n", r["err"])
