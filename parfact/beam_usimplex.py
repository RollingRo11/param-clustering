"""Fit v2 WITH the U-simplex and compute its keep-top-k curves (canonical
under logodds + oracle)."""
import json
import sys
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))

@function(name="v2-usimplex", image=IMG, gpu="L40S", pool="ondemand-l40s",
          cpu=8, memory="32Gi", timeout=2400, retries=0,
          disks=[DurableDisk(name="cofact-c600", size="10Gi",
                             mount_path="/cofact")])
def run():
    import subprocess as sp, torch
    from pathlib import Path
    p = sp.run([sys.executable, "-u", "run_parfact_v2.py",
                "--v1", "/cofact/runs/B_layer_K1200_C600",
                "--objective", "idiv", "--u_simplex", "--s_simplex",
                "--ckpt", "induction_model_100k.pt"],
               capture_output=True, text=True)
    if p.returncode:
        return json.dumps({"fit_error": p.stderr[-500:]})
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
    run_dir = Path("/cofact/runs/B_layer_K1200_C600_v2_idiv_usimplex_ssimplex")
    fact = torch.load(run_dir / "factorization.pt", weights_only=False,
                      map_location=dev)
    V = fact["V"].to(dev)
    matrices = sorted(set(fact["atom_matrix"]), key=fact["atom_matrix"].index)
    basis = AtomBasis.build(model, matrices, fact["config"]["variant"])
    comps = {n: t.to(dev) for n, t in basis.components(V).items()}
    z = collect_attributions(model, basis, seq, pos, y, score="logodds") @ V
    blob = curves_for(model, comps, z.abs(), events, dev,
                      orders=("canonical", "oracle"), seed=0)
    blob["fit_tail"] = p.stdout.strip().splitlines()[-6:]
    return json.dumps(blob)

if __name__ == "__main__":
    r = run.remote()
    if r:
        import pathlib
        pathlib.Path("out/curves_v2/curves_v2_usimplex.json").write_text(r)
        print("saved")
    else:
        print("FAILED")
