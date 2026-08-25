"""Sweep lambda_v (V-row entropy pruning) on the U-simplex config; run
keep-top-k curves (logodds canonical + oracle) for fits with R2 > 0.95."""
import json
import sys
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))

@function(name="v2-lv-sweep", image=IMG, gpu="L40S", pool="ondemand-l40s",
          cpu=8, memory="32Gi", timeout=3600, retries=0,
          disks=[DurableDisk(name="cofact-c600", size="10Gi",
                             mount_path="/cofact")])
def run():
    import subprocess as sp, torch
    from pathlib import Path
    out = {}
    kept = []
    for lv in ("0.00001", "0.00003", "0.0001"):
        p = sp.run([sys.executable, "-u", "run_parfact_v2.py",
                    "--v1", "/cofact/runs/B_layer_K1200_C600",
                    "--objective", "idiv", "--u_simplex",
                    "--lambda_v", lv, "--ckpt", "induction_model_100k.pt"],
                   capture_output=True, text=True)
        tail = p.stdout.strip().splitlines()[-8:]
        out[lv] = {"rc": p.returncode, "tail": tail,
                   "err": p.stderr[-300:] if p.returncode else ""}
        if p.returncode == 0:
            import json as j
            mtxt = Path("/cofact/runs/B_layer_K1200_C600_v2_idiv_usimplex"
                        f"_lv{float(lv):g}/metrics.json").read_text()
            m = j.loads(mtxt)
            out[lv]["metrics"] = m
            if m["r2_attr_euclid"] > 0.95:
                kept.append(lv)

    from induction_model import InductionModel
    from atoms import AtomBasis, collect_attributions, make_events
    from curves_v2 import curves_for
    dev = "cuda:0"
    model = InductionModel().to(dev)
    model.load_state_dict(torch.load("induction_model_100k.pt")["state_dict"])
    model.eval()
    [q.requires_grad_(False) for q in model.parameters()]
    events = make_events(model, 2048, "final", seed=1000)
    seq, pos, y = events["seq"], events["pos"], events["y"]
    curves = {}
    for lv in kept:
        run_dir = Path("/cofact/runs/B_layer_K1200_C600_v2_idiv_usimplex"
                       f"_lv{float(lv):g}")
        fact = torch.load(run_dir / "factorization.pt", weights_only=False,
                          map_location=dev)
        V = fact["V"].to(dev)
        matrices = sorted(set(fact["atom_matrix"]),
                          key=fact["atom_matrix"].index)
        basis = AtomBasis.build(model, matrices, fact["config"]["variant"])
        comps = {n: t.to(dev) for n, t in basis.components(V).items()}
        z = collect_attributions(model, basis, seq, pos, y,
                                 score="logodds") @ V
        curves[lv] = curves_for(model, comps, z.abs(), events, dev,
                                orders=("canonical", "oracle"), seed=0)
    return json.dumps({"fits": out, "curves": curves})

if __name__ == "__main__":
    r = run.remote()
    if r:
        import pathlib
        pathlib.Path("out/curves_v2/curves_lv_sweep.json").write_text(r)
        print("SAVED")
    else:
        print("FAILED")
