"""Full Table-1-style census (per matrix x token type + unique counts) for
the fully paper-exact concat-gate run spd_C600_concat."""
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch==2.7.1", "matplotlib", "numpy"]))

@function(name="concat-census", image=IMG, gpu="L40S",
          pool="ondemand-l40s", cpu=4, memory="16Gi", timeout=1200,
          retries=0,
          disks=[DurableDisk(name="spd-c600-concat", size="10Gi",
                             mount_path="/spd")])
def census():
    import torch, torch.nn as nn
    from pathlib import Path
    from induction_model import InductionModel, gen_batch
    from spd_toy_concat import MODULES, MatrixGate, install
    dev = "cuda:0"
    state = torch.load("/spd/runs/spd_C600_concat/spd_state.pt",
                       weights_only=True, map_location=dev)
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
    seq, s_pos, m_tok = gen_batch(2048, dev, gen)
    B, S = seq.shape
    rows = torch.arange(B, device=dev)
    m_pos = s_pos + 1
    rand = torch.randint(4, S - 5, (B,), device=dev, generator=gen)
    coll = (rand == s_pos) | (rand == m_pos)
    rand = torch.where(coll, (rand + 2) % (S - 5) + 4, rand)
    pos_of = {"s1": s_pos, "m": m_pos,
              "s2": torch.full_like(s_pos, S - 1), "random": rand}
    out = {}
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        model(seq)
        total = 0
        for nm in sorted(MODULES):
            g = gates[nm.replace(".", "_")](wrappers[nm].last_input)[0]
            act = g > 0.5
            row = {t: round(float(act[rows, pos_of[t]].float().sum(-1)
                                  .mean()), 3) for t in pos_of}
            uniq = torch.nonzero(act.any(0).any(0)).squeeze(-1)
            row["unique"] = int(uniq.numel())
            total += int(uniq.numel())
            out[nm] = row
        out["total_unique"] = total

        # exact loss terms on this eval batch (stochastic terms averaged
        # over 32 mask draws), correctly labeled for Table-1 panel (c)
        from spd_toy_concat import (faithfulness_loss, masked_kl,
                                    importance_minimality_loss)
        target_logits = model(seq)
        g_lo, g_up = {}, {}
        for nm in MODULES:
            lo, up = gates[nm.replace(".", "_")](wrappers[nm].last_input)
            g_lo[nm], g_up[nm] = lo, up
        ones = {n: torch.ones_like(g) for n, g in g_lo.items()}
        out["loss_faithful"] = float(faithfulness_loss(wrappers))
        out["loss_recon_plain"] = float(masked_kl(
            model, wrappers, seq, target_logits, ones))
        st, lw = 0.0, 0.0
        for _ in range(32):
            masks = {n: g + (1 - g) * torch.rand_like(g)
                     for n, g in g_lo.items()}
            st += float(masked_kl(model, wrappers, seq, target_logits, masks))
            lw += sum(float(masked_kl(model, wrappers, seq, target_logits,
                                      masks, subset=[n]))
                      for n in MODULES) / len(MODULES)
        out["loss_stoch_recon_avg32"] = st / 32
        out["loss_stoch_recon_layerwise_avg32"] = lw / 32
        out["loss_imp_min_p0.1"] = float(
            importance_minimality_loss(g_up, 0.1))
    return out

if __name__ == "__main__":
    import json
    r = census.remote()
    print(json.dumps(r, indent=1) if r else "FAILED")
    if r:
        import pathlib
        pathlib.Path("out/concat_census.json").write_text(json.dumps(r))
