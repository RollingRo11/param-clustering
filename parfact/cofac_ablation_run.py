"""Drive cofac_ablation.py on Beam (cofact disk + serverless RTX4090)."""
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))

@function(name="cofac-ablation", image=IMG, gpu="RTX4090", cpu=4,
          memory="16Gi", timeout=1800, retries=0,
          disks=[DurableDisk(name="cofact-c600", size="10Gi",
                             mount_path="/cofact")])
def run():
    import subprocess, sys
    p = subprocess.run([sys.executable, "-u", "cofac_ablation.py",
                        "--cofac", "/cofact/runs/B_layer_K1200_C600",
                        "--ckpt", "induction_model_100k.pt",
                        "--device", "cuda:0"],
                       capture_output=True, text=True)
    return {"rc": p.returncode, "out": p.stdout, "err": p.stderr[-1500:]}

if __name__ == "__main__":
    r = run.remote()
    print(r["out"] if r else "FAILED: None")
    if r and r["rc"]:
        print("--- stderr ---\n", r["err"])
