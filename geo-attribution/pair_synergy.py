"""Causal test: do interacting component pairs actually interact?

qk_interactions.py says the QK circuit's interaction mass sits overwhelmingly
on pairs of DIFFERENT components. That is a statement about weights. The causal
version asks whether those pairs behave like a circuit:

    synergy = dCE(ablate both) - dCE(ablate q-side) - dCE(ablate k-side)

Positive synergy means the two ablations are superadditive — the pair does
something together that neither does alone, which is what "these components
form a circuit" has to mean operationally. Zero means the components are
causally independent and the weight-space interaction was epiphenomenal.

The ablation is deliberately narrow. For a pair (c on the query side, c' on the
key side) found at layer l, only c's mass in that layer's q_proj and only c''s
mass in that layer's k_proj are removed. Ablating whole components everywhere
would confound the QK claim with everything else the components do.

Control: pairs drawn at random from the same per-head candidate pools, so both
arms have the same mass profile and only the measured interaction differs.

    python3.12 pair_synergy.py --n_pairs 12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
from german_permatrix import PerMatrixEditor
from german_vpd_1b import log


def ce_of(fwd, idx, chunk=4):
    out = []
    for s in range(0, idx.shape[0], chunk):
        b = idx[s:s + chunk]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=True):
            lg = fwd(b)
        out.append(F.cross_entropy(
            lg[:, :-1].reshape(-1, lg.shape[-1]).float(),
            b[:, 1:].reshape(-1), reduction="none").view(b.shape[0], -1).mean(1))
    return float(torch.cat(out).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--data", default="wmdp_data.pt")
    ap.add_argument("--n_pairs", type=int, default=12)
    ap.add_argument("--n_blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pair_synergy.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    qk = json.loads((run_dir / "qk_interactions.json").read_text())
    arrays = torch.load(run_dir / "qk_interactions_arrays.pt",
                        weights_only=False)
    CQ, CK, LAY = arrays["cq"], arrays["ck"], arrays["layer"]

    # the same (layer, c, c') pair recurs across heads in top_pairs; keeping
    # duplicates would fake consistency and shrink the effective sample
    seen, top = set(), []
    for r in qk["top_pairs"]:
        if r["same"]:
            continue
        key = (r["layer"], r["q_component"], r["k_component"])
        if key in seen:
            continue
        seen.add(key)
        top.append(r)
        if len(top) >= args.n_pairs:
            break
    g = torch.Generator().manual_seed(args.seed)
    ctrl = []
    while len(ctrl) < args.n_pairs:
        n = int(torch.randint(0, CQ.shape[0], (1,), generator=g))
        a = int(torch.randint(0, CQ.shape[1], (1,), generator=g))
        b = int(torch.randint(0, CK.shape[1], (1,), generator=g))
        cq_, ck_ = int(CQ[n, a]), int(CK[n, b])
        key = (int(LAY[n]), cq_, ck_)
        if cq_ != ck_ and key not in seen:
            seen.add(key)
            ctrl.append({"layer": key[0], "q_component": cq_,
                         "k_component": ck_, "strength": None})

    data = torch.load(run_dir / args.data, weights_only=False,
                      map_location="cpu")
    idx = data["pile_eval"][:args.n_blocks].to(dev)

    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    target = geo1b.load_target_1b(dev)

    results = {"top": [], "control": []}
    for arm, pairs in (("top", top), ("control", ctrl)):
        for pi, rec in enumerate(pairs):
            l, cq_, ck_ = rec["layer"], rec["q_component"], rec["k_component"]
            t0 = time.perf_counter()
            ed = PerMatrixEditor(target, bank, [cq_, ck_], dev)
            mods = ed.modules
            qi = [i for i, m in enumerate(mods)
                  if m == f"hf.model.layers.{l}.self_attn.q_proj"]
            ki = [i for i, m in enumerate(mods)
                  if m == f"hf.model.layers.{l}.self_attn.k_proj"]

            def alpha_for(kill_q, kill_k):
                a = torch.ones(2, len(mods), device=dev)
                if kill_q:
                    a[0, qi] = 0.0          # slot 0 = cq_, its q_proj mass
                if kill_k:
                    a[1, ki] = 0.0          # slot 1 = ck_, its k_proj mass
                return a

            base = ce_of(lambda x: ed.logits(x, None), idx)
            d_q = ce_of(lambda x: ed.logits(x, alpha_for(True, False)), idx) - base
            d_k = ce_of(lambda x: ed.logits(x, alpha_for(False, True)), idx) - base
            d_b = ce_of(lambda x: ed.logits(x, alpha_for(True, True)), idx) - base
            ed.alpha = None
            syn = d_b - d_q - d_k
            row = {"layer": l, "q_component": cq_, "k_component": ck_,
                   "strength": rec["strength"], "base_ce": round(base, 5),
                   "d_q": round(d_q, 6), "d_k": round(d_k, 6),
                   "d_both": round(d_b, 6), "synergy": round(syn, 6),
                   "synergy_over_sum": round(
                       syn / max(abs(d_q) + abs(d_k), 1e-9), 4)}
            results[arm].append(row)
            log(f"{arm:<7} [{pi + 1}/{len(pairs)}] L{l} c{cq_}(q) x c{ck_}(k): "
                f"dq {d_q:+.5f} dk {d_k:+.5f} both {d_b:+.5f} -> "
                f"synergy {syn:+.6f}  ({time.perf_counter() - t0:.0f}s)")
            del ed
            torch.cuda.empty_cache()
    del bank

    summ = {}
    for arm in ("top", "control"):
        s = torch.tensor([r["synergy"] for r in results[arm]])
        rel = torch.tensor([r["synergy_over_sum"] for r in results[arm]])
        summ[arm] = {
            "n": len(s), "mean_synergy": round(float(s.mean()), 6),
            "median_synergy": round(float(s.median()), 6),
            "frac_positive": round(float((s > 0).float().mean()), 3),
            "mean_relative": round(float(rel.mean()), 4),
        }
    st = torch.tensor([r["synergy"] for r in results["top"]])
    sc = torch.tensor([r["synergy"] for r in results["control"]])
    # Welch t, small n, reported as a rough effect size not a p-value claim
    denom = (st.var() / len(st) + sc.var() / len(sc)).sqrt().clamp_min(1e-12)
    summ["top_minus_control_t"] = round(float((st.mean() - sc.mean()) / denom), 3)
    out = {"format": "pair_synergy_v1", "summary": summ, "pairs": results}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"\ntop     synergy mean {summ['top']['mean_synergy']:+.6f} "
        f"median {summ['top']['median_synergy']:+.6f} "
        f"({100 * summ['top']['frac_positive']:.0f}% positive)")
    log(f"control synergy mean {summ['control']['mean_synergy']:+.6f} "
        f"median {summ['control']['median_synergy']:+.6f} "
        f"({100 * summ['control']['frac_positive']:.0f}% positive)")
    log(f"t = {summ['top_minus_control_t']:+.2f}")
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
