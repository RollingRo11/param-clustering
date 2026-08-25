"""Frequency census for the concat-gate run (32-dim gate MLPs)."""
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))

@function(name="freq-census-concat", image=IMG, gpu="L40S",
          pool="ondemand-l40s", cpu=4, memory="16Gi", timeout=1200, retries=0,
          disks=[DurableDisk(name="spd-c600-concat", size="10Gi",
                             mount_path="/spd")])
def census():
    import torch, torch.nn as nn
    from pathlib import Path
    from induction_model import InductionModel, gen_batch
    from spd_toy_concat import MODULES, MatrixGate, install
    dev = "cuda:0"
    run = Path("/spd/runs/spd_C600_concat")
    state = torch.load(run / "spd_state.pt", weights_only=True,
                       map_location=dev)
    c_per = int(state["c_per_module"])
    model = InductionModel().to(dev)
    model.load_state_dict(torch.load("induction_model_100k.pt")["state_dict"])
    model.eval()
    wrappers = install(model, c_per)
    model.to(dev)
    for n, w in wrappers.items():
        w.V.data.copy_(state["wrappers"][n]["V"].to(dev))
        w.U.data.copy_(state["wrappers"][n]["U"].to(dev))
    gates = nn.ModuleDict({n.replace(".", "_"): MatrixGate(c_per)
                           for n in MODULES}).to(dev)
    gates.load_state_dict(state["gates"])

    gen = torch.Generator(device=dev).manual_seed(0)
    seq, s_pos, m_tok = gen_batch(8192, dev, gen)
    B, S = seq.shape
    rows = torch.arange(B, device=dev)
    m_pos = s_pos + 1
    out = {}
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        model(seq)
        for nm in ["layers.1.wv", "layers.0.wk"]:
            g = gates[nm.replace(".", "_")](wrappers[nm].last_input)[0]
            gm = g[rows, m_pos if nm == "layers.1.wv" else s_pos]
            act = gm > 0.5
            freq = act.float().mean(0)
            o = freq.argsort(descending=True)
            tot = act.sum().item()
            cov = torch.cumsum(act[:, o].float().sum(0), 0) / max(tot, 1)
            out[nm] = {
                "n_active_any": int((freq > 0).sum()),
                "avg_per_token": round(float(act.float().sum(1).mean()), 2),
                "top12_freq": [(int(i), round(float(freq[i]), 4))
                               for i in o[:12]],
                "coverage_at_k": {k: round(float(cov[k - 1]), 3)
                                  for k in (5, 11, 20, 50)},
                "n_freq_gt_10pct": int((freq > 0.10).sum()),
                "n_freq_gt_25pct": int((freq > 0.25).sum()),
                "n_freq_gt_50pct": int((freq > 0.50).sum())}
    return out

if __name__ == "__main__":
    import json
    r = census.remote()
    print(json.dumps(r, indent=1) if r else "FAILED")
