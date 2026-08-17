"""Compare the induction components found by co-factorization vs SPD.

Both setups agree the induction prediction runs on ~13-15 components; this
asks whether they are the SAME components (proposal sec 7):

  - parameter-space alignment: |cos| between component weight tensors
    (flattened over the 6 matrices), against a random-pair baseline;
  - usage alignment: Pearson correlation over 2048 induction events between
    co-fac's usage share |z_c| and each SPD subcomponent's gate strength
    (max g over the event's circuit positions s1, m, s2);
  - matrix footprint: where each co-fac pool component's weight mass lives
    vs the single matrix each matched SPD subcomponent occupies.

Pools: co-fac = components ever inside an induction event's 90% |z|-mass
set; SPD = subcomponents with g > 0.5 at a circuit position in >= 1% of
events.

    python compare_components.py
"""
import argparse
from pathlib import Path

import torch

from induction_model import InductionModel, gen_batch
from atoms import AtomBasis, collect_grads
from spd_analysis import load_spd
from spd_toy import MODULES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofac", type=Path,
                    default=Path(__file__).parent / "out/B_layer_K1200_C600")
    ap.add_argument("--spd", type=Path,
                    default=Path(__file__).parent / "out/spd_C600")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--n_seq", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gen = torch.Generator(device=dev).manual_seed(1000)
    seq, s_pos, m_tok = gen_batch(args.n_seq, dev, gen)
    B, S = seq.shape
    rows = torch.arange(B, device=dev)
    final = torch.full_like(s_pos, S - 1)

    # -- co-fac: components, usage shares at the induction event -------------
    fact = torch.load(args.cofac / "factorization.pt", weights_only=False,
                      map_location=dev)
    basis = AtomBasis.build(model, sorted(set(fact["atom_matrix"]),
                            key=fact["atom_matrix"].index),
                            fact["config"]["variant"])
    cf = basis.components(fact["V"].to(dev))         # name -> [600, o, i]
    with torch.no_grad():
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
    cf_pool = torch.nonzero(in90.any(0)).squeeze(-1)

    # -- SPD: subcomponents, gate strengths at circuit positions -------------
    smodel, wrappers, gates, c_per = load_spd(args.spd, args.ckpt, dev)
    sp = {n: t.float().to(dev) for n, t in torch.load(
        args.spd / "components.pt", weights_only=True,
        map_location=dev)["components"].items()}
    with torch.no_grad():
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        smodel(seq)
        g = torch.cat([gates[n.replace(".", "_")](wrappers[n].last_input)[0]
                       for n in sorted(MODULES)], dim=2)   # [B, S, 600]
    g_circ = torch.stack([g[rows, p] for p in (s_pos, s_pos + 1, final)]
                         ).amax(0)                          # [B, 600]
    freq = (g_circ > 0.5).float().mean(0)
    spd_pool = torch.nonzero(freq > 0.01).squeeze(-1)
    mat_of = {i: sorted(MODULES)[i // c_per] for i in range(600)}
    print(f"pools: co-fac {len(cf_pool)} components, SPD {len(spd_pool)} "
          f"subcomponents (g>0.5 at a circuit position in >1% of events)")

    # -- A: parameter-space cosine -------------------------------------------
    def flat(comps, ids):
        return torch.cat([comps[nm][ids].flatten(1) for nm in sorted(
            n for n in comps)], dim=1)
    A = flat(cf, cf_pool)
    Bm = flat(sp, spd_pool)
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Bm = Bm / Bm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (A @ Bm.T).abs()                               # [n_cf, n_spd]
    gen2 = torch.Generator(device=dev).manual_seed(7)
    rand_ids = torch.randint(0, 600, (64,), device=dev, generator=gen2)
    Rf = flat(sp, rand_ids)
    Rf = Rf / Rf.norm(dim=1, keepdim=True).clamp_min(1e-12)
    base = (A @ Rf.T).abs().mean().item()
    print(f"\nA) parameter-space |cos|: best-match mean "
          f"{cos.max(1).values.mean():.3f} (co-fac->SPD), "
          f"{cos.max(0).values.mean():.3f} (SPD->co-fac); "
          f"random-pair baseline {base:.3f}")

    # -- B: usage correlation -------------------------------------------------
    zs = share[:, cf_pool]                               # [B, n_cf]
    gs = g_circ[:, spd_pool]                             # [B, n_spd]
    zc = zs - zs.mean(0)
    gc = gs - gs.mean(0)
    corr = (zc.T @ gc) / (zc.norm(dim=0)[:, None]
                          * gc.norm(dim=0)[None] + 1e-9)
    print(f"B) usage corr over {B} events: best-match mean "
          f"{corr.max(1).values.mean():.3f} (co-fac->SPD), "
          f"{corr.max(0).values.mean():.3f} (SPD->co-fac)")

    # -- top matched pairs ----------------------------------------------------
    print("\ntop co-fac -> SPD matches (by usage corr; param |cos| shown):")
    mean_share = share[:, cf_pool].mean(0)
    top_cf = mean_share.argsort(descending=True)[:10]
    for i in top_cf.tolist():
        j = int(corr[i].argmax())
        cf_id = int(cf_pool[i])
        spd_id = int(spd_pool[j])
        mass = {nm.split(".weight")[0].replace("layers.", "L"):
                cf[nm][cf_id].pow(2).sum().item() for nm in cf}
        tot = sum(mass.values())
        foot = max(mass, key=mass.get)
        print(f"  cf {cf_id:3d} (share {mean_share[i]:.3f}, "
              f"{foot} {100 * mass[foot] / tot:.0f}% of mass) -> "
              f"spd {spd_id:3d} [{mat_of[spd_id].replace('layers.', 'L')}] "
              f"corr {corr[i, j]:+.2f}  |cos| {cos[i, j]:.3f}")

    # -- footprints -----------------------------------------------------------
    from collections import Counter
    cf_foot = Counter()
    for cf_id in cf_pool.tolist():
        mass = {nm: cf[nm][cf_id].pow(2).sum().item() for nm in cf}
        cf_foot[max(mass, key=mass.get).split(".weight")[0]] += 1
    spd_foot = Counter(mat_of[int(i)] for i in spd_pool)
    print("\nC) pool footprint by (dominant) matrix:")
    for nm in sorted(set(cf_foot) | set(spd_foot)):
        print(f"  {nm:<14} co-fac {cf_foot.get(nm, 0):3d}   "
              f"SPD {spd_foot.get(nm, 0):3d}")


if __name__ == "__main__":
    main()
