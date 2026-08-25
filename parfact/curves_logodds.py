"""Canonical keep-top-k curves for v1/v2 under the NON-SATURATING logodds
score (base CE ~1e-5 saturates grad log p, polluting logp-based z). Oracle
and SPD curves are unchanged from curves_v2.py runs."""
import json
import sys
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))

@function(name="curves-logodds", image=IMG, gpu="L40S",
          pool="ondemand-l40s", cpu=8, memory="32Gi", timeout=1800,
          retries=0,
          disks=[DurableDisk(name="cofact-c600", size="10Gi",
                             mount_path="/cofact")])
def run():
    import torch, math
    from pathlib import Path
    from induction_model import InductionModel
    from atoms import AtomBasis, collect_attributions, make_events
    from curves_v2 import curves_for
    dev = "cuda:0"
    model = InductionModel().to(dev)
    model.load_state_dict(torch.load("induction_model_100k.pt")["state_dict"])
    model.eval()
    [p.requires_grad_(False) for p in model.parameters()]
    events = make_events(model, 2048, "final", seed=1000)
    seq, pos, y = events["seq"], events["pos"], events["y"]
    out = {}
    for tag, run_dir in (("v1", "/cofact/runs/B_layer_K1200_C600"),
                         ("v2", "/cofact/runs/B_layer_K1200_C600_v2_idiv")):
        fact = torch.load(Path(run_dir) / "factorization.pt",
                          weights_only=False, map_location=dev)
        V = fact["V"].to(dev)
        matrices = sorted(set(fact["atom_matrix"]),
                          key=fact["atom_matrix"].index)
        basis = AtomBasis.build(model, matrices, fact["config"]["variant"])
        comps = {n: t.to(dev) for n, t in basis.components(V).items()}
        z = collect_attributions(model, basis, seq, pos, y,
                                 score="logodds") @ V
        blob = curves_for(model, comps, z.abs(), events, dev,
                          orders=("canonical",), seed=0)
        out[tag] = blob
    return json.dumps(out)

if __name__ == "__main__":
    r = run.remote()
    if r:
        import pathlib
        pathlib.Path("out/curves_v2/curves_logodds.json").write_text(r)
        print("saved out/curves_v2/curves_logodds.json")
    else:
        print("FAILED")
