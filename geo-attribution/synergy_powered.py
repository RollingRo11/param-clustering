"""Powered version: screen for components that matter, then pair them.

component_synergy.py could not settle whether co-firing predicts causal
interaction, for a reason visible in its own data: |synergy| tracks the SMALLER
of the two main effects (Spearman +0.54), and 27 of 40 pairs had a component
whose ablation moved CE by less than 0.001 nats. An interaction term cannot be
resolved when the main effects are at the noise floor, so most of the sample
carried no information and the comparison was decided by two outliers.

So: screen single-component ablations first, keep only components whose removal
is measurable, and draw both arms from that same pool — high-lift pairs against
low-lift pairs of the SAME components. Every main effect is then large by
construction and identical in distribution across arms, and lift is the only
thing that varies.

    python3.12 synergy_powered.py --screen 48 --keep 14 --n_pairs 18
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import geo1b  # noqa: F401
from german_permatrix import PerMatrixEditor
from german_vpd_1b import log
from pair_synergy import ce_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--screen", type=int, default=48)
    ap.add_argument("--keep", type=int, default=14)
    ap.add_argument("--n_pairs", type=int, default=18)
    ap.add_argument("--n_blocks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="synergy_powered.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    arr = torch.load(run_dir / "coactivation_arrays.pt", weights_only=False)
    Coact, marg = arr["coact"].to(dev), arr["marg"].to(dev)
    C = Coact.shape[0]
    lift = Coact / (marg[:, None] * marg[None, :]).clamp_min(1e-30)

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    idx = torch.cat([data["pile_eval"],
                     data["bio_retain_eval"]])[:args.n_blocks].to(dev)
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    target = geo1b.load_target_1b(dev)

    # ---- screen: most-active components are the ones with a chance of
    # moving CE at all, so rank candidates by marginal activation ----
    cand = torch.topk(marg, args.screen).indices.tolist()
    log(f"screening {len(cand)} components by single ablation")
    main_eff = {}
    for i, c in enumerate(cand):
        ed = PerMatrixEditor(target, bank, [c], dev)
        n = len(ed.modules)
        base = ce_of(lambda z: ed.logits(z, None), idx)
        d = ce_of(lambda z: ed.logits(z, torch.zeros(1, n, device=dev)),
                  idx) - base
        ed.alpha = None
        main_eff[c] = d
        del ed
        torch.cuda.empty_cache()
        if (i + 1) % 8 == 0:
            log(f"  screened {i + 1}/{len(cand)}")
    ranked = sorted(main_eff.items(), key=lambda kv: -abs(kv[1]))
    keep = [c for c, _ in ranked[:args.keep]]
    log("kept: " + ", ".join(f"c{c}({main_eff[c]:+.4f})" for c in keep))

    # ---- all pairs within the kept pool, split by lift ----
    allp = [(a, b) for i, a in enumerate(keep) for b in keep[i + 1:]]
    allp.sort(key=lambda p: -float(lift[p[0], p[1]]))
    k = min(args.n_pairs // 2, len(allp) // 2)
    arms = {"high_lift": allp[:k], "low_lift": allp[-k:]}
    log(f"{len(allp)} pairs in pool; high-lift range "
        f"{float(lift[arms['high_lift'][-1]]):.1f}..{float(lift[arms['high_lift'][0]]):.1f}, "
        f"low-lift range {float(lift[arms['low_lift'][-1]]):.3f}.."
        f"{float(lift[arms['low_lift'][0]]):.3f}")

    results = {a: [] for a in arms}
    for arm, ps in arms.items():
        for pi, (a, b) in enumerate(ps):
            ed = PerMatrixEditor(target, bank, [a, b], dev)
            n = len(ed.modules)

            def al(ka, kb):
                x = torch.ones(2, n, device=dev)
                if ka:
                    x[0, :] = 0.0
                if kb:
                    x[1, :] = 0.0
                return x

            base = ce_of(lambda z: ed.logits(z, None), idx)
            da = ce_of(lambda z: ed.logits(z, al(True, False)), idx) - base
            db = ce_of(lambda z: ed.logits(z, al(False, True)), idx) - base
            dd = ce_of(lambda z: ed.logits(z, al(True, True)), idx) - base
            ed.alpha = None
            syn = dd - da - db
            results[arm].append({
                "a": a, "b": b, "lift": round(float(lift[a, b]), 4),
                "d_a": round(da, 6), "d_b": round(db, 6),
                "d_both": round(dd, 6), "synergy": round(syn, 6),
                "relative": round(syn / max(abs(da) + abs(db), 1e-9), 4)})
            log(f"{arm:<10} [{pi + 1}/{len(ps)}] c{a} x c{b} lift "
                f"{float(lift[a, b]):8.2f}: da {da:+.4f} db {db:+.4f} "
                f"both {dd:+.4f} -> syn {syn:+.5f} rel "
                f"{results[arm][-1]['relative']:+.3f}")
            del ed
            torch.cuda.empty_cache()
    del bank

    summ = {}
    for arm in arms:
        rel = torch.tensor([r["relative"] for r in results[arm]])
        syn = torch.tensor([r["synergy"] for r in results[arm]])
        summ[arm] = {
            "n": len(rel),
            "mean_relative": round(float(rel.mean()), 4),
            "median_relative": round(float(rel.median()), 4),
            "sd_relative": round(float(rel.std()), 4),
            "frac_positive": round(float((rel > 0).float().mean()), 3),
            "mean_synergy": round(float(syn.mean()), 6),
        }
    a = torch.tensor([r["relative"] for r in results["high_lift"]])
    b = torch.tensor([r["relative"] for r in results["low_lift"]])
    den = (a.var() / len(a) + b.var() / len(b)).sqrt().clamp_min(1e-12)
    summ["welch_t_relative"] = round(float((a.mean() - b.mean()) / den), 3)
    out = {"format": "synergy_powered_v1",
           "main_effects": {str(c): round(v, 6) for c, v in main_eff.items()},
           "summary": summ, "pairs": results}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    for arm in arms:
        s = summ[arm]
        log(f"{arm:<10} relative synergy {s['mean_relative']:+.4f} "
            f"(median {s['median_relative']:+.4f}, sd {s['sd_relative']:.4f}, "
            f"{100 * s['frac_positive']:.0f}% positive, n={s['n']})")
    log(f"Welch t (relative) = {summ['welch_t_relative']:+.2f}")
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
