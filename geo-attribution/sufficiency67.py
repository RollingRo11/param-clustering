"""Per-input sufficiency: how few components suffice to compute ONE output token?

This is the metric that actually tests an attribution sensor. The minimality
curve in sensor_study67.py ranks components ONCE, globally, by mass
(sum of swgt * W^2) and deletes the lightest -- an input-independent ordering
that barely uses the sensor's per-input signal. Here the ordering is produced
per input, by the sensor, for a single target token:

  1. pick a held-out sequence and a target position t
  2. reward = logit[t, gold_{t+1}]  (one scalar, PRE-softmax)
  3. run the sensor's forward/backward for that reward
  4. attribution of weight entry (o,i) to the reward, i.e. the first-order
     effect of zeroing it:   A_oi = (sum_j g_o^(j) p_i^(j)) * W_oi
     For IG sensors A is averaged over the weight-scaling path.
  5. attribution of component c = sum_oi share_c[o,i] * A_oi, read straight off
     the softpart bank with a weighted bincount
  6. KEEP the top-k components, zero every other share:
     W' = W * sum_{c in topk} share_c
  7. cross-entropy of that one token under W'

Because the shares sum to 1 at every weight entry, k=C restores the model
exactly, so each curve must return to the unablated CE at its right edge --
a built-in correctness check on the whole bank.

    python3.12 sufficiency67.py --C 256 --seed 0
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
from sensor_study67 import MODULES, SENSORS, load67, capture, build_spec, kmeans


def log(m):
    print(f"[suff67] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--feat_dim", type=int, default=65536)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--kmeans_iters", type=int, default=25)
    ap.add_argument("--soft_s", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--n_samples", type=int, default=48,
                    help="held-out (sequence, target position) pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensors", default=",".join(SENSORS))
    ap.add_argument("--out", type=Path, default=Path("out/suff67.json"))
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))
    names = [s for s in args.sensors.split(",") if s]
    t00 = time.perf_counter()

    KEEP = [k for k in [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
                        256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
            if k < args.C] + [args.C]

    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    # stratified by meta.pile_set_name: equal block quota per subset.
    # Block shuffle fixed at seed 0, so --seed varies only the pilot
    # position sample and the k-means init, never the corpus.
    want_blocks = args.positions // 64 + args.eval_seqs + 8
    ids_cpu, blk_labels, pile_stats = load_pile_blocks(
        tok, want_blocks, args.seq, seed=0, tokenizer_name=S67.TOKENIZER)
    IDS = ids_cpu.to(dev)
    n_blk = IDS.shape[0]
    log(f"tokens {tuple(IDS.shape)}")

    fit_n = min(n_blk - args.eval_seqs, max(1, args.positions // 64))
    per_seq = min(args.seq - 8, max(1, args.positions // fit_n))
    fit_ids, eval_ids = IDS[:fit_n], IDS[-args.eval_seqs:]
    avail = torch.arange(4, args.seq - 2)

    # held-out (sequence, target position) pairs, identical across sensors
    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_samples)]

    results, base_tok_ce = {}, None
    for name in names:
        cfg = SENSORS[name]
        t_s = time.perf_counter()
        model = load67(dev, cfg["mode"])
        W0 = {p: model.get_submodule(p).weight.detach().clone() for p in MODULES}
        Wt = {p: model.get_submodule(p).weight for p in MODULES}
        K = cfg["K"]

        @torch.no_grad()
        def restore():
            for p in MODULES:
                Wt[p].copy_(W0[p])

        @torch.no_grad()
        def tok_ce(b, t):
            lg = model(eval_ids[b:b + 1])[0, t].float()
            return float(F.cross_entropy(lg[None], eval_ids[b, t + 1][None]))

        if base_tok_ce is None:
            base_tok_ce = float(np.mean([tok_ce(b, t) for b, t in samp]))
            log(f"unablated CE on the {len(samp)} target tokens: "
                f"{base_tok_ce:.4f}")

        # ---------- build the decomposition (same as sensor_study67) ----------
        gcpu = torch.Generator().manual_seed(args.seed)
        sels = [avail[torch.randperm(avail.numel(), generator=gcpu)[:per_seq]]
                for _ in range(0, fit_n, 8)]
        Ps, Gs = [], []
        path = cfg.get("ig_path", "weights")
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for p in MODULES:
                    Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
            Pk = {p: [] for p in MODULES}
            Gk = {p: [] for p in MODULES}
            for bi, s in enumerate(range(0, fit_n, 8)):
                b = fit_ids[s:s + 8]
                if b.shape[0] == 0:
                    continue
                sel = sels[bi].to(dev)
                emb = None
                if path == "inputs":
                    with torch.no_grad():
                        xc, xf = model.wte(b), model.wte(b.roll(1, 0))
                    emb = xf + a * (xc - xf)
                P, G = capture(model, b, embed=emb)
                for p in MODULES:
                    Pk[p].append(P[p][:, sel].reshape(-1, P[p].shape[-1]).half())
                    Gk[p].append(G[p][:, sel].reshape(-1, G[p].shape[-1]).half())
                del P, G
            Ps.append({p: torch.cat(v) for p, v in Pk.items()})
            Gs.append({p: torch.cat(v) for p, v in Gk.items()})
        restore()
        N = Ps[0][MODULES[0]].shape[0]
        gen = torch.Generator(device=dev).manual_seed(args.seed + 977)
        if cfg.get("drop_grad"):
            Gs = [{p: torch.ones_like(v) for p, v in g.items()} for g in Gs]
        if cfg.get("random"):
            Ps = [{p: torch.randn(v.shape, generator=gen, device=dev,
                                  dtype=v.dtype) for p, v in s.items()}
                  for s in Ps]
            Gs = [{p: torch.randn(v.shape, generator=gen, device=dev,
                                  dtype=v.dtype) for p, v in s.items()}
                  for s in Gs]

        p2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Ps]).mean(0)
               for p in MODULES}
        g2m = {p: torch.stack([s[p].float().pow(2).mean(0) for s in Gs]).mean(0)
               for p in MODULES}
        spec, scales, D = build_spec(model, args.feat_dim, args.seed, dev,
                                     p2m, g2m)
        X = torch.zeros(N, D, device=dev)
        off = 0
        for p in MODULES:
            r, c = spec[p]
            w = r.numel()
            acc = torch.zeros(N, w, device=dev)
            for st in range(K):
                acc += Gs[st][p].float()[:, r] * Ps[st][p].float()[:, c]
            X[:, off:off + w] = (acc / K) * scales[p][None]
            off += w
        X = X[:, :off].clamp(-6e4, 6e4)
        Xc = X - X.mean(0)
        q = min(X.shape[1], N, args.embed_dim + 64)
        gg = torch.Generator(device=dev).manual_seed(args.seed)
        Q = torch.linalg.qr(torch.randn(X.shape[1], q, generator=gg,
                                        device=dev))[0]
        for _ in range(4):
            Q = torch.linalg.qr(Xc.t() @ (Xc @ Q))[0]
        Z = Xc @ Q
        sm = (Z.t() @ Z) / N
        val, vec = torch.linalg.eigh(0.5 * (sm + sm.t()))
        E = Xc @ (Q @ vec[:, val.argsort(descending=True)[:args.embed_dim]]
                  ).contiguous()
        del Xc, Z, sm, val, vec, Q, X
        _, lab = kmeans(E, args.C, args.kmeans_iters, args.seed)
        M = torch.zeros(args.C, N, device=dev)
        M[lab, torch.arange(N, device=dev)] = 1.0
        M = M / M.sum(1, keepdim=True).clamp_min(1e-30)
        del E
        torch.cuda.empty_cache()

        nz = [torch.nonzero(M[j], as_tuple=True)[0] for j in range(args.C)]
        sidx, swgt = {}, {}
        with torch.no_grad():
            for p in MODULES:
                d_out, d_in = W0[p].shape
                Sn = args.soft_s
                vals = torch.zeros(Sn, d_out, d_in, device=dev)
                idxs = torch.zeros(Sn, d_out, d_in, dtype=torch.int16,
                                   device=dev)
                for c0 in range(0, args.C, 32):
                    cc = min(32, args.C - c0)
                    Mm = torch.zeros(cc, d_out, d_in, device=dev)
                    for st in range(K):
                        Pa = Ps[st][p].float().abs()
                        Ga = Gs[st][p].float().abs()
                        for j in range(cc):
                            r = nz[c0 + j]
                            if r.numel() == 0:
                                continue
                            Mm[j] += (Ga[r] * M[c0 + j, r][:, None]).t() @ Pa[r]
                    allv = torch.cat([vals, Mm / K])
                    alli = torch.cat([idxs, torch.arange(
                        c0, c0 + cc, device=dev, dtype=torch.int16
                    )[:, None, None].expand(-1, d_out, d_in)])
                    top, si = allv.topk(Sn, dim=0)
                    vals, idxs = top, alli.gather(0, si)
                w = vals.clamp_min(0)
                tot = w.sum(0, keepdim=True)
                w = w / tot.clamp_min(1e-30)
                w[0] = torch.where(tot[0] > 0, w[0], torch.ones_like(w[0]))
                sidx[p], swgt[p] = idxs, w.half()
        del Ps, Gs, M, nz
        torch.cuda.empty_cache()

        # ---------------- per-input sufficiency ----------------
        rg = torch.Generator(device=dev).manual_seed(args.seed + 31)
        curves = np.zeros((len(samp), len(KEEP)))
        curves_rand = np.zeros((len(samp), len(KEEP)))
        frac_kept = np.zeros((len(samp), len(KEEP)))
        for si_, (b, t) in enumerate(samp):
            restore()
            # attribution of THIS token, averaged over the sensor's path
            A = {p: torch.zeros_like(W0[p]) for p in MODULES}
            b_cf = (b + 1) % eval_ids.shape[0]
            for step in range(K):
                a = (step + 1) / K
                with torch.no_grad():
                    for p in MODULES:
                        Wt[p].copy_(W0[p] * (a if path == "weights" else 1.0))
                emb = None
                if path == "inputs":
                    with torch.no_grad():
                        xc = model.wte(eval_ids[b:b + 1])
                        xf = model.wte(eval_ids[b_cf:b_cf + 1])
                    emb = xf + a * (xc - xf)
                pre, post, hs = {}, {}, []
                for p in MODULES:
                    mod = model.get_submodule(p)

                    def hook(m, inp, out, _p=p):
                        pre[_p] = inp[0]
                        out.retain_grad()
                        post[_p] = out
                    hs.append(mod.register_forward_hook(hook))
                lg = model(eval_ids[b:b + 1], embed=emb)
                for h in hs:
                    h.remove()
                reward = lg[0, t, eval_ids[b, t + 1]].float()
                grads = torch.autograd.grad(reward, [post[p] for p in MODULES])
                for p, g in zip(MODULES, grads):
                    # sum over positions of g_o * p_i, times W_oi
                    A[p] += (g[0].float().t() @ pre[p][0].float()) / K
                del pre, post, grads, lg
            restore()
            for p in MODULES:
                A[p] *= W0[p]

            attr = torch.zeros(args.C, device=dev, dtype=torch.float64)
            with torch.no_grad():
                for p in MODULES:
                    attr += torch.bincount(
                        sidx[p].reshape(-1).int(),
                        weights=(swgt[p].float() * A[p][None]
                                 ).reshape(-1).double(), minlength=args.C)
            del A

            for tag, order, store in (
                    ("attr", torch.argsort(attr, descending=True), curves),
                    ("rand", torch.randperm(args.C, generator=rg, device=dev),
                     curves_rand)):
                rank = torch.empty(args.C, dtype=torch.int32, device=dev)
                rank[order] = torch.arange(args.C, dtype=torch.int32,
                                           device=dev)
                with torch.no_grad():
                    R = {p: rank[sidx[p].int()] for p in MODULES}
                    for ki, kk in enumerate(KEEP):
                        tot_kept = tot_all = 0.0
                        for p in MODULES:
                            keep = (swgt[p] * (R[p] < kk)).sum(0,
                                                               dtype=torch.float32)
                            Wt[p].copy_(W0[p] * keep)
                            if tag == "attr":
                                tot_kept += float((keep * W0[p].abs()).sum())
                                tot_all += float(W0[p].abs().sum())
                        store[si_, ki] = tok_ce(b, t)
                        if tag == "attr":
                            frac_kept[si_, ki] = tot_kept / max(tot_all, 1e-30)
                    del R
            restore()

        mean_c = curves.mean(0)
        mean_r = curves_rand.mean(0)
        # smallest k whose mean CE is within 0.05 nats of unablated
        suff = next((k for k, v in zip(KEEP, mean_c)
                     if v - base_tok_ce <= 0.05), args.C)
        results[name] = {
            "mode": cfg["mode"], "ig_k": K, "keep_grid": KEEP,
            "ce_attr": [round(float(v), 5) for v in mean_c],
            "ce_random": [round(float(v), 5) for v in mean_r],
            "frac_weight_kept": [round(float(v), 5)
                                 for v in frac_kept.mean(0)],
            "ce_attr_sem": [round(float(v), 5)
                            for v in curves.std(0, ddof=1) / np.sqrt(len(samp))],
            "sufficient_k_within_0.05": suff,
            "roundtrip_err_at_C": round(float(mean_c[-1] - base_tok_ce), 6)}
        log(f"{name:<18} K={K}  sufficient k={suff:>4}/{args.C}  "
            f"CE@k=16 {mean_c[list(KEEP).index(16)]:.3f}  "
            f"(rand {mean_r[list(KEEP).index(16)]:.3f})  "
            f"roundtrip {results[name]['roundtrip_err_at_C']:+.2e}  "
            f"({time.perf_counter() - t_s:.0f}s)")
        del model, W0, Wt, sidx, swgt
        torch.cuda.empty_cache()

    out = {"format": "suff67_v1", "model": "VPD 4L-Pile 67M target",
           "C": args.C, "seed": args.seed, "n_samples": len(samp),
           "base_token_ce": round(base_tok_ce, 5), "keep_grid": KEEP,
           "pile": pile_stats,
           "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log(f"wrote {args.out}  (total {time.perf_counter() - t00:.0f}s)")


if __name__ == "__main__":
    main()
