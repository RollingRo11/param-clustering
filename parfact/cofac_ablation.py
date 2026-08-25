"""Circuit-group ablations for the co-fac decomposition, mirroring the SPD
table in spd_analysis.py: same events (seed 1000), same clean_forward, same
groups (L0 prev-token / L1 match / L1 copy), same metrics (acc, m->s1, s2->m).

"Active" co-fac components = the induction pool from compare_components.py
(components ever inside an event's 90% |z|-mass set). SPD subcomponents each
live in ONE matrix; co-fac components span all six, so ablation is run two
ways: (a) whole components assigned to a group by dominant-mass matrix, and
(b) only the group's matrix blocks of every pool component (exact SPD analog).
"""
import argparse
import math
from pathlib import Path

import torch

from induction_model import InductionModel, gen_batch
from atoms import AtomBasis, collect_grads
from spd_analysis import clean_forward
from spd_toy import MODULES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofac", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gen = torch.Generator(device=dev).manual_seed(1000)   # match compare_components
    seq, s_pos, m_tok = gen_batch(args.n_seq, dev, gen)
    B, S = seq.shape
    rows = torch.arange(B, device=dev)
    m_pos = s_pos + 1
    final = torch.full_like(s_pos, S - 1)

    fact = torch.load(args.cofac / "factorization.pt", weights_only=False,
                      map_location=dev)
    basis = AtomBasis.build(model, sorted(set(fact["atom_matrix"]),
                            key=fact["atom_matrix"].index),
                            fact["config"]["variant"])
    cf = basis.components(fact["V"].to(dev))              # name -> [600, o, i]
    n_comp = next(iter(cf.values())).shape[0]
    sdkey = lambda nm: nm if nm.endswith(".weight") else nm + ".weight"

    # -- pool: same 90% |z|-mass rule as compare_components ------------------
    with torch.no_grad():
        y = model(seq)[rows, -1].argmax(-1)
    grads = collect_grads(model, list(cf), seq, final, y)
    z = torch.zeros(B, n_comp, device=dev)
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

    # -- dominant matrix per pool component ----------------------------------
    mass = torch.stack([cf[nm].flatten(1).square().sum(1) for nm in
                        sorted(cf)], dim=1)              # [600, 6]
    dom = mass.argmax(1)                                 # index into sorted(cf)
    names = sorted(cf)
    by_mat = {nm: pool[(dom[pool] == i)] for i, nm in enumerate(names)}
    print(f"pool: {len(pool)} components; by dominant matrix: "
          + ", ".join(f"{nm.replace('.weight','')}={len(by_mat[nm])}"
                      for nm in names))

    # -- clean reference -----------------------------------------------------
    sd = {k: v.detach().clone().to(dev) for k, v in
          torch.load(args.ckpt)["state_dict"].items()}
    logits, probs, _ = clean_forward(sd, seq)
    acc = (logits[rows, -1].argmax(-1) == m_tok).float().mean().item()
    a0 = probs[0][rows, m_pos, s_pos].mean().item()
    a1 = probs[1][rows, -1, m_pos].mean().item()
    print(f"\nclean model: acc {acc:.4f}; m->s1 (L0) {a0:.3f}; "
          f"s2->m (L1) {a1:.3f}")

    def evaluate(sd2):
        lg, pr, _ = clean_forward(sd2, seq)
        return ((lg[rows, -1].argmax(-1) == m_tok).float().mean().item(),
                pr[0][rows, m_pos, s_pos].mean().item(),
                pr[1][rows, -1, m_pos].mean().item())

    groups = {
        "L0 K+Q+V (prev-token step)": ["layers.0.wk", "layers.0.wq",
                                       "layers.0.wv"],
        "L1 Q+K (match step)": ["layers.1.wq", "layers.1.wk"],
        "L1 V (copy step)": ["layers.1.wv"],
    }
    match = lambda nm, mats: any(nm.startswith(m) for m in mats)

    print("\n(a) ablating WHOLE pool components grouped by dominant matrix:")
    for label, mats in groups.items():
        ids = torch.cat([by_mat[nm] for nm in names if match(nm, mats)])
        sd2 = {k: v.clone() for k, v in sd.items()}
        for nm in cf:                                    # full component: all blocks
            sd2[sdkey(nm)] -= cf[nm][ids].sum(0)
        a, x0, x1 = evaluate(sd2)
        print(f"  {label:<28} n={len(ids):3d}  acc {a:.4f}  "
              f"m->s1 {x0:.3f}  s2->m {x1:.3f}")

    print("\n(b) ablating only the group's MATRIX BLOCKS of all pool "
          "components (exact SPD analog):")
    for label, mats in groups.items():
        sd2 = {k: v.clone() for k, v in sd.items()}
        for nm in cf:
            if match(nm, mats):
                sd2[sdkey(nm)] -= cf[nm][pool].sum(0)
        a, x0, x1 = evaluate(sd2)
        print(f"  {label:<28} n={len(pool):3d}  acc {a:.4f}  "
              f"m->s1 {x0:.3f}  s2->m {x1:.3f}")


if __name__ == "__main__":
    main()
