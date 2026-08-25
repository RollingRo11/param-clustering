"""Run spd_analysis.py (Christensen-style census) on the finished SPD run."""
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))

@function(name="spd-analysis", image=IMG, gpu="RTX4090", cpu=4, memory="16Gi",
          disks=[DurableDisk(name="spd-c600", size="10Gi", mount_path="/spd")],
          timeout=1800, retries=0)
def analyse(run: str = "spd_C600", thresh: float = 0.5):
    import subprocess, sys
    p = subprocess.run([sys.executable, "-u", "spd_analysis.py",
                        "--run", f"/spd/runs/{run}",
                        "--ckpt", "induction_model_100k.pt",
                        "--thresh", str(thresh), "--device", "cuda:0"],
                       capture_output=True, text=True)
    return {"rc": p.returncode, "out": p.stdout, "err": p.stderr[-1200:]}

if __name__ == "__main__":
    r = analyse.remote()
    print(r["out"] if r else "None")
    if r and r["rc"]:
        print("--- stderr ---\n", r["err"])
