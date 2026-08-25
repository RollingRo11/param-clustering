"""Control: what fraction of each matrix's weight does each ablation remove?
||sum_{c in group} C_c[matrix]||_F / ||W||_F for co-fac pool and SPD actives."""
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))

@function(name="mass-check", image=IMG, gpu="RTX4090", cpu=4, memory="16Gi",
          timeout=1200, retries=0,
          disks=[DurableDisk(name="spd-c600", size="10Gi", mount_path="/spd"),
                 DurableDisk(name="cofact-c600", size="10Gi",
                             mount_path="/cofact")])
def check():
    import torch, json
    from pathlib import Path
    from induction_model import InductionModel, gen_batch
    from atoms import AtomBasis, collect_grads
    dev = "cuda:0"
    model = InductionModel().to(dev)
    ck = torch.load("induction_model_100k.pt")
    model.load_state_dict(ck["state_dict"])
    model.eval()
    [p.requires_grad_(False) for p in model.parameters()]
    sd = {k: v.to(dev) for k, v in ck["state_dict"].items()}

    gen = torch.Generator(device=dev).manual_seed(1000)
    seq, s_pos, m_tok = gen_batch(2048, dev, gen)
    B, S = seq.shape
    rows = torch.arange(B, device=dev)
    final = torch.full_like(s_pos, S - 1)

    out = {}
    # ---- co-fac pool, same construction as cofac_ablation.py ----
    fact = torch.load("/cofact/runs/B_layer_K1200_C600/factorization.pt",
                      weights_only=False, map_location=dev)
    basis = AtomBasis.build(model, sorted(set(fact["atom_matrix"]),
                            key=fact["atom_matrix"].index),
                            fact["config"]["variant"])
    cf = basis.components(fact["V"].to(dev))
    y = model(seq)[rows, -1].argmax(-1)
    grads = collect_grads(model, list(cf), seq, final, y)
    z = torch.zeros(B, 600, device=dev)
    for nm in cf:
        z += torch.einsum("noi,coi->nc", grads[nm], cf[nm])
    share = z.abs() / z.abs().sum(1, keepdim=True).clamp_min(1e-12)
    order = share.argsort(1, descending=True)
    cum = torch.gather(share, 1, order).cumsum(1)
    keep = cum < 0.90
    keep[:, 0] = True
    in90 = torch.zeros_like(keep)
    in90.scatter_(1, order, keep)
    pool = torch.nonzero(in90.any(0)).squeeze(-1)
    sk = lambda nm: nm if nm.endswith(".weight") else nm + ".weight"
    for nm in sorted(cf):
        W = sd[sk(nm)]
        out[f"cofac {nm.replace('.weight','')}"] = round(
            (cf[nm][pool].sum(0).norm() / W.norm()).item(), 4)

    # ---- SPD actives (unique anywhere, g>0.5), from analysis.json ----
    sp = torch.load("/spd/runs/spd_C600/components.pt", weights_only=True,
                    map_location=dev)["components"]
    mats = sorted(n.replace(".weight", "") for n in sp)
    from spd_analysis import load_spd
    smodel, wrappers, gates, c_per = load_spd(Path("/spd/runs/spd_C600"),
                                              Path("induction_model_100k.pt"), dev)
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        smodel(seq)                     # populate wrappers[.].last_input
        uniq = {}
        for nm in mats:
            g = gates[nm.replace(".", "_")](wrappers[nm].last_input)[0]
            uniq[nm] = torch.nonzero((g > 0.5).any(0).any(0)).squeeze(-1)
    for i, nm in enumerate(mats):
        W = sd[nm + ".weight"]
        ids = uniq[nm].to(dev)
        blk = sp[nm + ".weight" if nm + ".weight" in sp else nm].float().to(dev)
        # components.pt stores [600, o, i] with global ids; slice this matrix's
        gl = ids + i * c_per if blk.shape[0] == 600 else ids
        out[f"spd   {nm}"] = round((blk[gl].sum(0).norm() / W.norm()).item(), 4)
    return out

if __name__ == "__main__":
    r = check.remote()
    if r is None:
        print("FAILED")
    else:
        for k, v in r.items():
            print(f"{k:24} removed-mass fraction {v}")
