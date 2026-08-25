"""Fit the v2 (residual + I-divergence) co-factorization and compute
keep-top-k ablation curves for v1 / v2 / SPD on Beam.

    python beam_v2.py fit       # v2 fit (idiv) + euclid ablation fit
    python beam_v2.py curves    # canonical+oracle curves for all 3 methods
    python beam_v2.py fetch     # pull the curve jsons back locally
"""
import sys
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))
DISKS = [DurableDisk(name="cofact-c600", size="10Gi", mount_path="/cofact"),
         DurableDisk(name="spd-c600", size="10Gi", mount_path="/spd")]
GPU = dict(image=IMG, gpu="L40S", pool="ondemand-l40s", cpu=8,
           memory="32Gi", disks=DISKS, timeout=-1, retries=0)

V1 = "/cofact/runs/B_layer_K1200_C600"


@function(name="v2-fit", **GPU)
def fit():
    import subprocess as sp
    outs = {}
    for obj in ("idiv", "euclid"):
        p = sp.run([sys.executable, "-u", "run_parfact_v2.py",
                    "--v1", V1, "--objective", obj,
                    "--ckpt", "induction_model_100k.pt"],
                   capture_output=True, text=True)
        outs[obj] = {"rc": p.returncode,
                     "tail": p.stdout.strip().splitlines()[-10:],
                     "err": p.stderr[-400:]}
    return outs


@function(name="v2-curves", **GPU)
def curves():
    import subprocess as sp, os
    p = sp.run([sys.executable, "-u", "curves_v2.py",
                "--v1", V1, "--v2", V1 + "_v2_idiv",
                "--spd", "/spd/runs/spd_C600",
                "--ckpt", "induction_model_100k.pt",
                "--out_dir", "/cofact/runs/curves_v2"],
               capture_output=True, text=True)
    return {"rc": p.returncode, "tail": p.stdout.strip().splitlines()[-12:],
            "err": p.stderr[-600:]}


@function(image=Image(python_version="python3.12"), cpu=1, memory="2Gi",
          disks=DISKS, timeout=300, retries=0)
def fetch():
    import os, base64
    out = {}
    d = "/cofact/runs/curves_v2"
    for f in os.listdir(d) if os.path.isdir(d) else []:
        out[f] = base64.b64encode(open(os.path.join(d, f), "rb").read()
                                  ).decode()
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    r = {"fit": fit, "curves": curves, "fetch": fetch}[cmd].remote()
    if cmd == "fetch":
        import base64, pathlib
        pathlib.Path("out/curves_v2").mkdir(parents=True, exist_ok=True)
        for name, blob in (r or {}).items():
            pathlib.Path(f"out/curves_v2/{name}").write_bytes(
                base64.b64decode(blob))
            print("fetched", name)
    else:
        import json
        print(json.dumps(r, indent=1) if r else "FAILED: None")
