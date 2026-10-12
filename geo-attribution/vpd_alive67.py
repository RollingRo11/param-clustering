"""Alive subcomponents of the VPD checkpoint, counted on our own corpora.

VPD's 38,912 rank-one subcomponents include a large dead fraction (the paper
reports ~10,000 alive, mean causal importance > 1e-6).  For fraction-of-units
comparisons the denominator should be the alive count, measured here on the
same corpora the measured-ownership pipeline uses:

    python -u vpd_alive67.py --labs lab.pt lab_1m.pt --out out/vpd_alive67.json

Per subcomponent: mean causal importance over tokens (hard-sigmoid clamp of
the CI logit to [0, 1], and the lower-leaky variant with slope 0.01 below 0),
and the fraction of tokens on which it is active (logit > 0).  Per token: how
many subcomponents are active.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import sensor_study67 as S67
from sensor_study67 import MODULES, load67

if os.environ.get("COFAC_DATA"):
    S67.CKPT = Path(os.environ["COFAC_DATA"]) / "target" / "model_step_99999.pt"

import compare_vpd as CV  # noqa: E402

THRESH = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]


def log(m):
    print(f"[vpd_alive] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--labs", nargs="+", required=True,
                    help="lab*.pt corpus files (ids [N, T])")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--out", default="out/vpd_alive67.json")
    ap.add_argument("--component-stats", type=Path,
                    help="Optionally save per-component means and activity frequencies as JSON")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    model = load67(dev, "plain")
    DATA = Path(os.environ["COFAC_DATA"])
    sd = torch.load(str(DATA / "vpd" / "model_400000.pth"), map_location="cpu",
                    weights_only=False)
    order_m = sorted(MODULES)
    key = {m: "_components." + m.replace(".", "-") for m in MODULES}
    split = [sd[key[n] + ".U"].shape[0] for n in order_m]
    ci_net = CV.CiFn({k: v.float().to(dev) for k, v in sd.items()
                      if k.startswith("ci_fn")}, order_m, split).to(dev).eval()
    n_sub = sum(split)
    offs = np.cumsum([0] + split)
    log(f"VPD: {n_sub} subcomponents; per module {dict(zip(order_m, split))}")

    @torch.no_grad()
    def ci_logits(idx):
        pre, hs = {}, []
        for m in MODULES:
            mod = model.get_submodule(m)

            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
            hs.append(mod.register_forward_hook(hook))
        model(idx)
        for h in hs:
            h.remove()
        lg = ci_net({m: pre[m] for m in MODULES})
        return torch.cat([lg[m] for m in order_m], -1)      # [B, T, n_sub]

    out = {"format": "vpd_alive67_v1", "n_sub": n_sub, "thresholds": THRESH,
           "modules": order_m, "split": split, "corpora": {}}
    component_stats = {"checkpoint": str(DATA / "vpd" / "model_400000.pth"),
                       "definition": "mean(clamp(CI logit, 0, 1)) > 1e-6 over all corpus tokens",
                       "modules": order_m, "split": split, "corpora": {}}
    for lf in args.labs:
        lab = torch.load(lf, map_location="cpu", weights_only=False)
        ids = lab["ids"].long().to(dev)
        N, T = ids.shape
        s_clamp = torch.zeros(n_sub, device=dev, dtype=torch.float64)
        s_leaky = torch.zeros(n_sub, device=dev, dtype=torch.float64)
        n_act = torch.zeros(n_sub, device=dev, dtype=torch.float64)
        per_tok = []
        t0 = time.perf_counter()
        for c0 in range(0, N, args.chunk):
            lg = ci_logits(ids[c0:c0 + args.chunk]).float()
            s_clamp += lg.clamp(0, 1).double().sum((0, 1))
            s_leaky += torch.where(lg < 0, 0.01 * lg, lg.clamp(max=1)).double().sum((0, 1))
            act = lg > 0
            n_act += act.double().sum((0, 1))
            per_tok.append(act.sum(-1).flatten().cpu())
            del lg, act
            if (c0 // args.chunk + 1) % 16 == 0:
                log(f"{Path(lf).stem}: {min(c0 + args.chunk, N)}/{N} sequences "
                    f"({time.perf_counter() - t0:.0f}s)")
        n_tok = N * T
        mc, ml, fa = s_clamp / n_tok, s_leaky / n_tok, n_act / n_tok
        per_tok = torch.cat(per_tok).float()
        name = Path(lf).stem
        res = {"n_tokens": n_tok,
               "alive_mean_ci_clamp": {f"{t:g}": int((mc > t).sum()) for t in THRESH},
               "alive_mean_ci_leaky": {f"{t:g}": int((ml > t).sum()) for t in THRESH},
               "ever_active": int((n_act > 0).sum()),
               "active_on_at_least_1pct_tokens": int((fa > 0.01).sum()),
               "per_module_alive_1e-6": {m: int((mc[offs[i]:offs[i + 1]] > 1e-6).sum())
                                         for i, m in enumerate(order_m)},
               "active_per_token": {"mean": float(per_tok.mean()),
                                    "quantiles": {q: float(torch.quantile(per_tok, q))
                                                  for q in (0.1, 0.5, 0.9, 0.99)}},
               "mean_ci_quantiles_over_subcomponents": {
                   q: float(torch.quantile(mc.float(), q)) for q in (0.5, 0.75, 0.9, 0.99)}}
        out["corpora"][name] = res
        if args.component_stats:
            component_stats["corpora"][name] = {
                "path": str(lf), "n_tokens": n_tok,
                "mean_clamped_ci": mc.cpu().tolist(),
                "active_token_fraction": fa.cpu().tolist(),
            }
        log(f"{name}: {n_tok} tokens ({time.perf_counter() - t0:.0f}s); alive (mean CI > 1e-6) "
            f"{res['alive_mean_ci_clamp']['1e-06']} / {n_sub}; ever active {res['ever_active']}; "
            f"active per token mean {res['active_per_token']['mean']:.0f}, median "
            f"{res['active_per_token']['quantiles'][0.5]:.0f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}")
    if args.component_stats:
        args.component_stats.parent.mkdir(parents=True, exist_ok=True)
        args.component_stats.write_text(json.dumps(component_stats))
        log(f"wrote {args.component_stats}")


if __name__ == "__main__":
    main()
