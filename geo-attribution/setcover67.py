"""Set-cover-style refinement of weight ownership, post-hoc.

The bank assigns each weight entry shares over its top-S co-activation owners.
Per-token sufficiency keeps the top-k components by attribution; an entry is
recovered only insofar as its owners are kept. This refinement reweights each
entry's shares toward the candidate owners that RANK EARLY on the tokens where
that entry actually matters -- one coordinate-descent step of minimising
expected per-token coverage (the greedy set-cover direction), using pilot
tokens only, never the eval set.

For refinement token t (M of them, drawn from fit sequences):
  A_t        = per-entry attribution (IG path), [d_out, d_in] per module
  s_t[c]     = component scores via the bank, rank_t = descending rank
  gain_t[c]  = 1 / (rank_t[c] + r0)      early owners are worth more
  acc[j,o,i] += |A_t[o,i] * W[o,i]| * gain_t[sidx[j,o,i]]

new shares = normalise(swgt * acc)  (entries whose acc is all-zero keep their
original shares, so shares still sum to 1 everywhere and the k=C round-trip
stays exact).

Evaluated against the unrefined bank on the same 48 held-out tokens.

    python3.12 setcover67.py --C 4096 --refine_tokens 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sensor_study67 as S67
from sensor_study67 import MODULES, SENSORS, load67
from combine67 import build_bank


def log(m):
    print(f"[setcover] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--C", type=int, default=4096)
    ap.add_argument("--sensor", default="ig5")
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--feat_dim", type=int, default=65536)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--refine_tokens", type=int, default=256)
    ap.add_argument("--rank_r0", type=float, default=16.0)
    ap.add_argument("--refine_iters", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/setcover67.json"))
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))
    t00 = time.perf_counter()
    cfg = SENSORS[args.sensor]
    K, path = cfg["K"], cfg.get("ig_path", "weights")

    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    want = args.positions // 64 + args.eval_seqs + 8
    ids_cpu, _, _ = load_pile_blocks(tok, want, args.seq, seed=0,
                                     tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    fit_n = min(IDS.shape[0] - args.eval_seqs, max(1, args.positions // 64))
    fit_ids, eval_ids = IDS[:fit_n], IDS[-args.eval_seqs:]
    avail = torch.arange(4, args.seq - 2)
    gcpu = torch.Generator().manual_seed(args.seed)
    per_seq = min(args.seq - 8, max(1, args.positions // fit_n))
    sels = [avail[torch.randperm(avail.numel(), generator=gcpu)[:per_seq]]
            for _ in range(0, fit_n, 8)]

    model = load67(dev, cfg["mode"])
    W0 = {m: model.get_submodule(m).weight.detach().clone() for m in MODULES}
    Wt = {m: model.get_submodule(m).weight for m in MODULES}
    sidx, swgt = build_bank(model, cfg, W0, Wt, fit_ids, fit_n, sels, args, dev)
    log(f"bank built ({time.perf_counter()-t00:.0f}s)")

    C = args.C

    def token_attr_A(ids_row, t):
        """Per-entry attribution A*W for one (sequence, position)."""
        A = {m: torch.zeros_like(W0[m]) for m in MODULES}
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for m in MODULES:
                    Wt[m].copy_(W0[m] * (a if path == "weights" else 1.0))
            pre, post, hs = {}, {}, []
            for m in MODULES:
                mod = model.get_submodule(m)

                def hook(mm, inp, out, _m=m):
                    pre[_m] = inp[0]
                    out.retain_grad()
                    post[_m] = out
                hs.append(mod.register_forward_hook(hook))
            lg = model(ids_row[None])
            for h in hs:
                h.remove()
            reward = lg[0, t, ids_row[t + 1]].float()
            grads = torch.autograd.grad(reward, [post[m] for m in MODULES])
            for m, g in zip(MODULES, grads):
                A[m] += (g[0].float().t() @ pre[m][0].float()) / K
            del pre, post, grads, lg
        with torch.no_grad():
            for m in MODULES:
                Wt[m].copy_(W0[m])
        return {m: A[m] * W0[m] for m in MODULES}

    def comp_scores(AW, si, sw):
        v = torch.zeros(C, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for m in MODULES:
                v += torch.bincount(si[m].reshape(-1).int(),
                                    weights=(sw[m].float() * AW[m][None]
                                             ).reshape(-1).double(),
                                    minlength=C)
        return v

    # ---- refinement pass (fit tokens only) ----
    t1 = time.perf_counter()
    gref = torch.Generator().manual_seed(args.seed + 555)
    ref = [(int(torch.randint(0, fit_n, (1,), generator=gref)),
            int(torch.randint(64, args.seq - 2, (1,), generator=gref)))
           for _ in range(args.refine_tokens)]
    AWs = []
    for i, (b, t) in enumerate(ref):
        AWs.append({m: v.half() for m, v in token_attr_A(fit_ids[b], t).items()})
    log(f"cached {len(ref)} refinement attributions "
        f"({time.perf_counter()-t1:.0f}s)")
    sidx2 = sidx
    swgt2 = {m: swgt[m].clone() for m in MODULES}
    for it in range(args.refine_iters):
        acc = {m: torch.zeros_like(swgt[m], dtype=torch.float32)
               for m in MODULES}
        for AW in AWs:
            s = comp_scores(AW, sidx2, swgt2)
            order = torch.argsort(s, descending=True)
            rank = torch.empty(C, device=dev, dtype=torch.float32)
            rank[order] = torch.arange(C, device=dev, dtype=torch.float32)
            gain = 1.0 / (rank + args.rank_r0)
            with torch.no_grad():
                for m in MODULES:
                    acc[m] += AW[m].float().abs()[None] * gain[sidx[m].int()]
        with torch.no_grad():
            for m in MODULES:
                w = swgt[m].float() * acc[m]
                tot = w.sum(0, keepdim=True)
                w = torch.where(tot <= 0, swgt[m].float(),
                                w / tot.clamp_min(1e-30))
                swgt2[m] = w.half()
        log(f"refine iter {it+1}/{args.refine_iters} done "
            f"({time.perf_counter()-t1:.0f}s)")
    del AWs

    # ---- evaluation on held-out tokens, both banks ----
    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_samples)]
    KEEP = [k for k in [0, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768,
                        1024, 1536, 2048, 3072] if k < C] + [C]

    @torch.no_grad()
    def tok_ce(b, t):
        lg = model(eval_ids[b:b + 1])[0, t].float()
        return float(F.cross_entropy(lg[None], eval_ids[b, t + 1][None]))

    base = float(np.mean([tok_ce(b, t) for b, t in samp]))
    log(f"unablated CE on {len(samp)} eval tokens: {base:.4f}")
    results = {}
    for name, si, sw in (("original", sidx, swgt), ("refined", sidx2, swgt2)):
        curves = np.zeros((len(samp), len(KEEP)))
        for j, (b, t) in enumerate(samp):
            AW = token_attr_A(eval_ids[b], t)
            s = comp_scores(AW, si, sw)
            del AW
            order = torch.argsort(s, descending=True)
            rank = torch.empty(C, dtype=torch.int32, device=dev)
            rank[order] = torch.arange(C, dtype=torch.int32, device=dev)
            with torch.no_grad():
                R = {m: rank[si[m].int()] for m in MODULES}
                for ki, kk in enumerate(KEEP):
                    for m in MODULES:
                        keep = (sw[m].float() * (R[m] < kk)
                                ).sum(0, dtype=torch.float32)
                        Wt[m].copy_(W0[m] * keep)
                    curves[j, ki] = tok_ce(b, t)
                del R
            with torch.no_grad():
                for m in MODULES:
                    Wt[m].copy_(W0[m])
        mu = curves.mean(0)
        thr = next((k for k, v in zip(KEEP, mu) if v - base <= 0.25), C)
        results[name] = {
            "ce": [round(float(v), 5) for v in mu],
            "k_within_0.25": thr,
            "roundtrip_err_at_C": round(float(mu[-1] - base), 6)}
        log(f"{name:<9} k needed {thr:>5}/{C}  "
            f"CE@64 {mu[KEEP.index(64)]:.2f}  CE@256 {mu[KEEP.index(256)]:.2f}"
            f"  roundtrip {results[name]['roundtrip_err_at_C']:+.1e}")

    out = {"format": "setcover67_v1", "C": C, "sensor": args.sensor,
           "soft_s": args.soft_s, "refine_tokens": len(ref),
           "rank_r0": args.rank_r0, "n_samples": len(samp),
           "base_token_ce": round(base, 5), "keep_grid": KEEP,
           "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out} (total {time.perf_counter()-t00:.0f}s)")


if __name__ == "__main__":
    main()
