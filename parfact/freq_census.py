"""Per-subcomponent activation FREQUENCY census for L1.wv: is our 95 a flat
smear, or ~11 frequent components plus a rare tail (which would match the
paper under a frequency-based definition of active)?"""
from beam import Image, function
from beta9.type import DurableDisk

import sys
DISK = sys.argv[1] if len(sys.argv) > 1 else "spd-c600"
RUN = sys.argv[2] if len(sys.argv) > 2 else "spd_C600"
IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))

@function(name="freq-census", image=IMG, gpu="L40S", pool="ondemand-l40s",
          cpu=4, memory="16Gi", timeout=1200, retries=0,
          env={"CENSUS_RUN": RUN},
          disks=[DurableDisk(name=DISK, size="10Gi", mount_path="/spd")])
def census():
    import torch
    from pathlib import Path
    from induction_model import InductionModel, gen_batch
    from spd_analysis import load_spd
    dev = "cuda:0"
    import os
    run_dir = Path("/spd/runs") / os.environ["CENSUS_RUN"]
    smodel, wrappers, gates, c_per = load_spd(run_dir,
                                              Path("induction_model_100k.pt"), dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    seq, s_pos, m_tok = gen_batch(8192, dev, gen)
    B, S = seq.shape
    rows = torch.arange(B, device=dev)
    m_pos = s_pos + 1
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        smodel(seq)
        out = {}
        for nm in ["layers.1.wv", "layers.0.wk"]:
            g = gates[nm.replace(".", "_")](wrappers[nm].last_input)[0]
            gm = g[rows, m_pos if nm == "layers.1.wv" else s_pos]  # [B, C]
            act = gm > 0.5
            freq = act.float().mean(0)                             # [C]
            o = freq.argsort(descending=True)
            top = [(int(i), round(float(freq[i]), 4)) for i in o[:20]]
            tot = act.sum().item()
            cov = torch.cumsum(act[:, o].float().sum(0), 0) / max(tot, 1)
            kcov = {k: round(float(cov[k - 1]), 3) for k in (5, 11, 20, 50)}
            out[nm] = {"n_active_any": int((freq > 0).sum()),
                       "avg_per_token": round(float(act.float().sum(1).mean()), 2),
                       "top20_freq": top, "coverage_at_k": kcov,
                       "n_freq_gt_10pct": int((freq > 0.10).sum()),
                       "n_freq_gt_25pct": int((freq > 0.25).sum()),
                       "n_freq_gt_50pct": int((freq > 0.50).sum())}
        return out

if __name__ == "__main__":
    r = census.remote()
    import json
    print(json.dumps(r, indent=1) if r else "FAILED: None")
