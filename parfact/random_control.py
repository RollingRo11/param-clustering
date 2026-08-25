"""Matched random-ablation control for both methods: for each circuit group,
ablate the SAME COUNT of randomly chosen components from the same matrices
(3 seeds), vs the methods' own active/pool sets."""
from beam import Image, function
from beta9.type import DurableDisk

IMG = (Image(python_version="python3.12")
       .add_python_packages(["torch", "matplotlib", "numpy"]))

@function(name="random-control", image=IMG, gpu="RTX4090", cpu=4,
          memory="16Gi", timeout=1800, retries=0,
          disks=[DurableDisk(name="spd-c600", size="10Gi", mount_path="/spd"),
                 DurableDisk(name="cofact-c600", size="10Gi",
                             mount_path="/cofact")])
def control():
    import torch
    from pathlib import Path
    from induction_model import InductionModel, gen_batch
    from atoms import AtomBasis, collect_grads
    from spd_analysis import clean_forward, load_spd
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
    m_pos, final = s_pos + 1, torch.full_like(s_pos, S - 1)

    def evaluate(sd2):
        lg, pr, _ = clean_forward(sd2, seq)
        return ((lg[rows, -1].argmax(-1) == m_tok).float().mean().item(),
                pr[0][rows, m_pos, s_pos].mean().item(),
                pr[1][rows, -1, m_pos].mean().item())

    groups = {"L0": ["layers.0.wk", "layers.0.wq", "layers.0.wv"],
              "L1QK": ["layers.1.wq", "layers.1.wk"],
              "L1V": ["layers.1.wv"]}
    lines = []

    # ---- SPD: active sets per matrix, then matched random ------------------
    smodel, wrappers, gates, c_per = load_spd(Path("/spd/runs/spd_C600"),
                                              Path("induction_model_100k.pt"), dev)
    sp = {n: t.float().to(dev) for n, t in torch.load(
        "/spd/runs/spd_C600/components.pt", weights_only=True,
        map_location=dev)["components"].items()}
    mats = sorted(n.replace(".weight", "") for n in sp)
    off = {nm: i * c_per for i, nm in enumerate(mats)}
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        smodel(seq)
        act = {nm: torch.nonzero((gates[nm.replace(".", "_")](
            wrappers[nm].last_input)[0] > 0.5).any(0).any(0)).squeeze(-1)
            for nm in mats}

    def spd_ablate(idsmap):
        sd2 = {k: v.clone() for k, v in sd.items()}
        for nm, ids in idsmap.items():
            if len(ids):
                sd2[nm + ".weight"] -= sp[nm + ".weight"][ids + off[nm]].sum(0)
        return evaluate(sd2)

    for gname, gm in groups.items():
        a = spd_ablate({nm: act[nm] for nm in gm})
        rnd = []
        for s in range(3):
            g2 = torch.Generator(device=dev).manual_seed(100 + s)
            rnd.append(spd_ablate({nm: torch.randperm(c_per, device=dev,
                       generator=g2)[:len(act[nm])] for nm in gm}))
        r = torch.tensor(rnd).mean(0)
        lines.append(f"SPD   {gname:5} active acc {a[0]:.3f} "
                     f"(m->s1 {a[1]:.2f} s2->m {a[2]:.2f}) | "
                     f"random-k acc {r[0]:.3f} ({r[1]:.2f} {r[2]:.2f})")

    # ---- co-fac: pool by dominant matrix, then matched random --------------
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
    names = sorted(cf)
    mass = torch.stack([cf[nm].flatten(1).square().sum(1) for nm in names], 1)
    dom = mass.argmax(1)
    sk = lambda nm: nm if nm.endswith(".weight") else nm + ".weight"

    def cf_ablate(ids):
        sd2 = {k: v.clone() for k, v in sd.items()}
        for nm in cf:
            sd2[sk(nm)] -= cf[nm][ids].sum(0)
        return evaluate(sd2)

    for gname, gm in groups.items():
        gi = [i for i, nm in enumerate(names)
              if any(nm.startswith(m) for m in gm)]
        ids = pool[torch.isin(dom[pool], torch.tensor(gi, device=dev))]
        a = cf_ablate(ids)
        rnd = []
        for s in range(3):
            g2 = torch.Generator(device=dev).manual_seed(200 + s)
            rnd.append(cf_ablate(torch.randperm(600, device=dev,
                                                generator=g2)[:len(ids)]))
        r = torch.tensor(rnd).mean(0)
        lines.append(f"cofac {gname:5} pool(n={len(ids):2d}) acc {a[0]:.3f} "
                     f"(m->s1 {a[1]:.2f} s2->m {a[2]:.2f}) | "
                     f"random-k acc {r[0]:.3f} ({r[1]:.2f} {r[2]:.2f})")
    return "\n".join(lines)

if __name__ == "__main__":
    print(control.remote())
