"""VPD's keep AND drop curves on the SAME held-out tokens as stream67's eval.

The existing VPD numbers (klkeep_vpd_big*.json) were measured on the cofac
holdout events, while the geo numbers come from stream67's own eval sample, so
the two were only ever comparable in kind. This replicates stream67.phase_bank's
token selection exactly -- same pile blocks, same seed, same generator, same 48
(sequence, position) pairs -- and evaluates VPD's 38,912 rank-1 subcomponents on
them, so the curves are paired.

It also adds the necessity arm VPD never had: keep the top-k by causal
importance (sufficiency), and separately remove the top-k and keep the rest
(necessity), plus random-k nulls for both.

    python3 vpd_match67.py --out out/vpd_match67.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sensor_study67 as S67
from sensor_study67 import MODULES, load67

if os.environ.get("COFAC_DATA"):
    import pile_data
    S67.CKPT = Path(os.environ["COFAC_DATA"]) / "target" / "model_step_99999.pt"
    pile_data.CACHE = Path(os.environ["COFAC_DATA"]) / "cofac67" / "piledata"

import compare_vpd as CV  # noqa: E402  (needs S67.CKPT set first)


def log(m):
    print(f"[vpd_match] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--C_ref", type=int, default=8192,
                    help="geo component count the percent grid is taken from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/vpd_match67.json")
    args = ap.parse_args()
    dev = args.device

    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_cpu, _, _ = load_pile_blocks(tok, 168, args.seq, seed=0,
                                     tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    eval_ids = IDS[-args.eval_seqs:]
    # identical to stream67.phase_bank
    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_samples)]
    log(f"{len(samp)} eval tokens (same selection as stream67 phase_bank)")

    model = load67(dev, "plain")
    params = {p: model.get_submodule(p).weight for p in MODULES}
    DATA = Path(os.environ["COFAC_DATA"])
    sd = torch.load(str(DATA / "vpd" / "model_400000.pth"), map_location="cpu",
                    weights_only=False)
    order_m = sorted(MODULES)
    key = {m: "_components." + m.replace(".", "-") for m in MODULES}
    Uv = {m: sd[key[m] + ".U"].float().to(dev) for m in MODULES}
    Vv = {m: sd[key[m] + ".V"].float().to(dev) for m in MODULES}
    split = [Uv[n].shape[0] for n in order_m]
    ci_net = CV.CiFn({k: v.float().to(dev) for k, v in sd.items()
                      if k.startswith("ci_fn")}, order_m, split).to(dev).eval()
    n_sub = sum(split)
    offs, o = {}, 0
    for m in order_m:
        offs[m] = o
        o += Uv[m].shape[0]
    log(f"VPD: {n_sub} rank-1 subcomponents over {len(MODULES)} matrices")

    W0 = {p: params[p].data.clone() for p in MODULES}

    @torch.no_grad()
    def restore():
        for p in MODULES:
            params[p].data.copy_(W0[p])

    @torch.no_grad()
    def ci_at(b, t):
        pre, hs = {}, []
        for m in MODULES:
            mod = model.get_submodule(m)

            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
            hs.append(mod.register_forward_hook(hook))
        model(eval_ids[b:b + 1])
        for h in hs:
            h.remove()
        logits = ci_net({m: pre[m] for m in MODULES})
        return torch.cat([logits[m][0, t] for m in order_m])

    @torch.no_grad()
    def forward_with(keep_mask, b, t):
        """Subtract every subcomponent NOT in keep_mask, then read position t."""
        for m in order_m:
            sl = slice(offs[m], offs[m] + Uv[m].shape[0])
            rm = ~keep_mask[sl]
            if rm.any():
                params[m].data -= (Vv[m][:, rm] @ Uv[m][rm]).T
        lg = model(eval_ids[b:b + 1])[0, t].float()
        restore()
        return lg

    # the geo keep grid, expressed as percentages, so the x axes line up
    GEO_K = [k for k in (4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024,
                         1536, 2048, 3072, 4096, 6144) if k < args.C_ref]
    pcts = [100.0 * k / args.C_ref for k in GEO_K] + [100.0]
    ks = [max(1, int(round(n_sub * p / 100))) for p in pcts]

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    acc = {p: {"keep": [], "drop": [], "keep_rand": [], "drop_rand": []}
           for p in pcts}
    t0 = time.perf_counter()
    with torch.no_grad():
        for j, (b, t) in enumerate(samp):
            base = F.log_softmax(model(eval_ids[b:b + 1])[0, t].float(), -1)
            order = ci_at(b, t).argsort(descending=True)
            rnd = torch.randperm(n_sub, device=dev, generator=gen)

            def kl(mask):
                return float(F.kl_div(F.log_softmax(forward_with(mask, b, t), -1),
                                      base, log_target=True, reduction="sum"))
            for p, k in zip(pcts, ks):
                top = torch.zeros(n_sub, dtype=torch.bool, device=dev)
                top[order[:k]] = True
                rtop = torch.zeros(n_sub, dtype=torch.bool, device=dev)
                rtop[rnd[:k]] = True
                acc[p]["keep"].append(kl(top))
                acc[p]["drop"].append(kl(~top))
                acc[p]["keep_rand"].append(kl(rtop))
                acc[p]["drop_rand"].append(kl(~rtop))
            if (j + 1) % 8 == 0:
                log(f"{j+1}/{len(samp)} tokens "
                    f"({(j+1)/(time.perf_counter()-t0):.2f}/s)")
    restore()

    out = {"format": "vpd_match67_v1", "n_sub": n_sub, "C_ref": args.C_ref,
           "n_samples": len(samp), "geo_k": GEO_K, "pct": pcts, "vpd_k": ks,
           "eval": "stream67 phase_bank selection (168 blocks, seed 0, "
                   "last 32 seqs, generator 12345)",
           "results": {kk: [round(float(np.mean(acc[p][kk])), 5) for p in pcts]
                       for kk in ("keep", "drop", "keep_rand", "drop_rand")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}")
    for p, k, a, d in zip(pcts, ks, out["results"]["keep"],
                          out["results"]["drop"]):
        log(f"  {p:6.2f}% ({k:>6} subcomps)  keep {a:7.3f}   drop {d:7.3f}")


if __name__ == "__main__":
    main()
