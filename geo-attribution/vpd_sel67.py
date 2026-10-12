"""VPD subcomponent selectivity, measured the way ours is.

For a sample of alive subcomponents: delete the rank-one piece, run the
labelled corpus, and compare KL on the tokens where the subcomponent is
active (causal-importance logit > 0, its own gate) with KL on all other
tokens.  Null: another alive subcomponent of similar mean importance, judged
on the same token split.

    python -u vpd_sel67.py --lab lab.pt --out out/vpd_sel67.json
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
    S67.CKPT = Path(os.environ["COFAC_DATA"]) / "target" / "model_step_99999.pt"

import compare_vpd as CV  # noqa: E402


def log(m):
    print(f"[vpd_sel] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lab", required=True)
    ap.add_argument("--n_seqs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--n_top", type=int, default=96)
    ap.add_argument("--n_rand", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/vpd_sel67.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    model = load67(dev, "plain")
    params = {p: model.get_submodule(p).weight for p in MODULES}
    W0 = {p: params[p].data.clone() for p in MODULES}
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
    offs = np.cumsum([0] + split)
    mod_of = np.concatenate([[i] * n for i, n in enumerate(split)])
    lab = torch.load(args.lab, map_location="cpu", weights_only=False)
    ids = lab["ids"].long()[:args.n_seqs].to(dev)
    N, T = ids.shape
    V = model.cfg["vocab_size"]

    @torch.no_grad()
    def ci_logits(idx):
        pre, hs = {}, []
        for m in MODULES:
            mod = model.get_submodule(m)

            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
            hs.append(mod.register_forward_hook(hook))
        lg = model(idx)
        for h in hs:
            h.remove()
        c = ci_net({m: pre[m] for m in MODULES})
        return lg, torch.cat([c[m] for m in order_m], -1)

    # base distribution + activity of every subcomponent on every token
    t0 = time.perf_counter()
    base_lp = torch.empty(N, T, V, device=dev)
    active = torch.zeros(n_sub, N * T, dtype=torch.bool, device=dev)
    mean_ci = torch.zeros(n_sub, device=dev, dtype=torch.float64)
    with torch.no_grad():
        for c0 in range(0, N, args.chunk):
            lg, c = ci_logits(ids[c0:c0 + args.chunk])
            base_lp[c0:c0 + args.chunk] = F.log_softmax(lg.float(), -1)
            act = (c > 0).reshape(-1, n_sub).t()          # [n_sub, chunk*T]
            active[:, c0 * T:c0 * T + act.shape[1]] = act
            mean_ci += c.float().clamp(0, 1).double().reshape(-1, n_sub).sum(0)
            del lg, c
    mean_ci /= N * T
    alive = (mean_ci > 1e-6).nonzero().flatten()
    log(f"{N}x{T} tokens; {alive.numel()} alive subcomponents; base + activity in "
        f"{time.perf_counter() - t0:.0f}s")

    @torch.no_grad()
    def kl_map():
        out = torch.empty(N, T, device=dev)
        for c0 in range(0, N, args.chunk):
            lp = F.log_softmax(model(ids[c0:c0 + args.chunk]).float(), -1)
            b = base_lp[c0:c0 + args.chunk]
            out[c0:c0 + args.chunk] = (b.exp() * (b - lp)).sum(-1)
            del lp
        return out.flatten()

    @torch.no_grad()
    def ablate(s):
        m = order_m[mod_of[s]]
        j = s - offs[mod_of[s]]
        # W [d_out, d_in]; U [C, d_out]?? no: vpd_match67 uses (V[:, rm] @ U[rm]).T,
        # i.e. V is [d_in, C] and U is [C, d_out], so the piece is (V[:, j] outer U[j]).T
        params[m].data -= torch.outer(Vv[m][:, j], Uv[m][j]).T

    @torch.no_grad()
    def restore():
        for p in MODULES:
            params[p].data.copy_(W0[p])

    # sanity on the ablation orientation: removing one piece must change the model
    g = torch.Generator().manual_seed(args.seed)
    top = alive[mean_ci[alive].argsort(descending=True)[:args.n_top]].tolist()
    rest = alive[torch.randperm(alive.numel(), generator=g)[:args.n_rand * 3]].tolist()
    rnd = [s for s in rest if s not in set(top)][:args.n_rand]
    chosen = top + rnd
    rows = []
    t0 = time.perf_counter()
    for i, s in enumerate(chosen):
        own = active[s]
        other = ~own
        if own.sum() < 4:
            continue
        ablate(s)
        K = kl_map()
        restore()
        # null: another alive subcomponent with the nearest mean importance
        cand = alive[(mean_ci[alive] - mean_ci[s]).abs().argsort()[1:8]]
        ns = int(cand[torch.randint(0, cand.numel(), (1,), generator=g)])
        ablate(ns)
        Kn = kl_map()
        restore()
        rows.append({"sub": int(s), "module": order_m[mod_of[s]], "mean_ci": float(mean_ci[s]),
                     "own_tokens": int(own.sum()), "own_frac": float(own.float().mean()),
                     "kl_own": float(K[own].mean()), "kl_other": float(K[other].mean()),
                     "kl_all": float(K.mean()), "null_sub": ns,
                     "null_kl_own": float(Kn[own].mean()), "null_kl_other": float(Kn[other].mean()),
                     "kind": "top" if i < len(top) else "random"})
        if (i + 1) % 32 == 0:
            log(f"{i + 1}/{len(chosen)} ({time.perf_counter() - t0:.0f}s)")
    def med(k, rr):
        return float(np.median([r[k] for r in rr])) if rr else None
    summ = {}
    for kind in ("all", "top", "random"):
        rr = rows if kind == "all" else [r for r in rows if r["kind"] == kind]
        if not rr:
            continue
        summ[kind] = {"n": len(rr),
                      "selectivity": float(np.median([r["kl_own"] / max(r["kl_other"], 1e-9) for r in rr])),
                      "null_selectivity": float(np.median([r["null_kl_own"] / max(r["null_kl_other"], 1e-9) for r in rr])),
                      "kl_own": med("kl_own", rr), "kl_other": med("kl_other", rr),
                      "own_frac": med("own_frac", rr)}
    out = {"format": "vpd_sel67_v1", "n_seqs": N, "n_alive": int(alive.numel()),
           "own_definition": "tokens where the subcomponent's causal-importance logit > 0",
           "summary": summ, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}: " + json.dumps(summ))


if __name__ == "__main__":
    main()
