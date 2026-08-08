"""Do components that FIRE together also interact CAUSALLY?

The QK synergy test in pair_synergy.py was underpowered: ablating one
component's mass in a single q_proj moves CE by ~0.0008 nats, which cannot
resolve an interaction term. This ablates whole components — all 112 matrices —
so the main effects are large enough for the interaction to be measurable.

    synergy = dCE(ablate both) - dCE(ablate a) - dCE(ablate b)

Selection is by LIFT, not raw co-activation. Raw co-activation just picks the
components that fire most often; lift = Coact(a,b) / (p(a) p(b)) picks pairs
that fire together more than their marginals predict, which is the actual
hypothesis "these two belong to one mechanism".

The control is a REWIRING null: the same components as the top-lift arm, paired
differently. That holds every marginal activation, every mass fraction and
every per-component main effect fixed, and varies only which partner each is
tested against — so a difference cannot come from the top arm containing
bigger or busier components.

    python3.12 component_synergy.py --n_pairs 20
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
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--n_blocks", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="0 ablates the owned mass; >1 amplifies it")
    ap.add_argument("--min_coact", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="component_synergy.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    arr = torch.load(run_dir / "coactivation_arrays.pt", weights_only=False)
    Coact, marg = arr["coact"].to(dev), arr["marg"].to(dev)
    C = Coact.shape[0]
    lift = Coact / (marg[:, None] * marg[None, :]).clamp_min(1e-30)
    mask = (Coact >= args.min_coact) & (~torch.eye(C, dtype=torch.bool,
                                                   device=dev))
    lift = torch.where(mask, lift, torch.zeros_like(lift))
    flat = torch.triu(lift, 1).flatten()
    top = torch.topk(flat, args.n_pairs)
    pairs = [(int(i) // C, int(i) % C) for i in top.indices]
    log(f"top-lift pairs, lift range {float(top.values[-1]):.1f} .. "
        f"{float(top.values[0]):.1f}")

    # rewiring null: same components, shuffled partners, no accidental re-pairs
    pool = [c for p in pairs for c in p]
    g = torch.Generator().manual_seed(args.seed)
    real = {tuple(sorted(p)) for p in pairs}
    ctrl = []
    tries = 0
    while len(ctrl) < args.n_pairs and tries < 10000:
        tries += 1
        i, j = torch.randint(0, len(pool), (2,), generator=g).tolist()
        a, b = pool[i], pool[j]
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in real or key in {tuple(sorted(c)) for c in ctrl}:
            continue
        ctrl.append((a, b))

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    blocks = torch.cat([data["pile_eval"], data["bio_retain_eval"]])
    idx = blocks[:args.n_blocks].to(dev)
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    target = geo1b.load_target_1b(dev)
    A = args.alpha

    results = {"top": [], "control": []}
    for arm, ps in (("top", pairs), ("control", ctrl)):
        for pi, (a, b) in enumerate(ps):
            t0 = time.perf_counter()
            ed = PerMatrixEditor(target, bank, [a, b], dev)
            n = len(ed.modules)

            def al(ka, kb):
                x = torch.ones(2, n, device=dev)
                if ka:
                    x[0, :] = A
                if kb:
                    x[1, :] = A
                return x

            base = ce_of(lambda z: ed.logits(z, None), idx)
            da = ce_of(lambda z: ed.logits(z, al(True, False)), idx) - base
            db = ce_of(lambda z: ed.logits(z, al(False, True)), idx) - base
            dd = ce_of(lambda z: ed.logits(z, al(True, True)), idx) - base
            ed.alpha = None
            syn = dd - da - db
            results[arm].append({
                "a": a, "b": b,
                "lift": round(float(lift[a, b]), 3),
                "coact": float(Coact[a, b]),
                "d_a": round(da, 6), "d_b": round(db, 6),
                "d_both": round(dd, 6), "synergy": round(syn, 6),
                "relative": round(syn / max(abs(da) + abs(db), 1e-9), 4)})
            log(f"{arm:<7} [{pi + 1}/{len(ps)}] c{a} x c{b} lift "
                f"{float(lift[a, b]):7.1f}: da {da:+.5f} db {db:+.5f} "
                f"both {dd:+.5f} -> synergy {syn:+.6f} "
                f"({time.perf_counter() - t0:.0f}s)")
            del ed
            torch.cuda.empty_cache()
    del bank

    summ = {}
    for arm in ("top", "control"):
        s = torch.tensor([r["synergy"] for r in results[arm]])
        m = torch.tensor([abs(r["d_a"]) + abs(r["d_b"])
                          for r in results[arm]])
        summ[arm] = {
            "n": len(s), "mean_synergy": round(float(s.mean()), 6),
            "median_synergy": round(float(s.median()), 6),
            "sd": round(float(s.std()), 6),
            "frac_positive": round(float((s > 0).float().mean()), 3),
            "mean_main_effect": round(float(m.mean()), 6),
            "mean_relative": round(float((s / m.clamp_min(1e-9)).mean()), 4),
        }
    st = torch.tensor([r["synergy"] for r in results["top"]])
    sc = torch.tensor([r["synergy"] for r in results["control"]])
    den = (st.var() / len(st) + sc.var() / len(sc)).sqrt().clamp_min(1e-12)
    summ["welch_t"] = round(float((st.mean() - sc.mean()) / den), 3)
    out = {"format": "component_synergy_v1", "alpha": A,
           "summary": summ, "pairs": results}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    for arm in ("top", "control"):
        s = summ[arm]
        log(f"{arm:<7} synergy {s['mean_synergy']:+.6f} +- {s['sd']:.6f} "
            f"(median {s['median_synergy']:+.6f}, "
            f"{100 * s['frac_positive']:.0f}% positive, main effect "
            f"{s['mean_main_effect']:.5f}, relative {s['mean_relative']:+.3f})")
    log(f"Welch t = {summ['welch_t']:+.2f}")
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
