"""Time SPD steps on the reserved H100 pool, and assert we actually got one."""
import subprocess, sys, time
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))
DISK = [DurableDisk(name="spd-c600", size="10Gi", mount_path="/data")]

@function(image=IMG, gpu="H100", pool="ondemand-h100", cpu=8, memory="32Gi",
          disks=DISK, timeout=900, retries=0)
def bench(steps: int = 4000):
    name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
    assert "H100" in name, f"expected H100, got {name}"   # fail fast on substitution
    t0 = time.time()
    subprocess.run([sys.executable, "-u", "spd_toy.py", "--c_per_module", "100",
                    "--steps", str(steps), "--ckpt", "induction_model_100k.pt",
                    "--out", "/data/runs/spd_bench", "--world", "1"], check=True)
    dt = time.time() - t0
    return {"gpu": name, "steps": steps, "seconds": round(dt, 1),
            "ms_per_step": round(1000 * dt / steps, 2)}

if __name__ == "__main__":
    print(bench.remote(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 4000))
