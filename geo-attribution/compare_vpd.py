"""How close does our decomposition get to VPD's, on the same tokens?

Both decompose the SAME 67M target (goodfire/spd t-9d2b8f02). Theirs: 38,912
rank-1 subcomponents (W_s = U_s outer V_s per module) gated by a learned
causal-importance transformer (run s-55ea3f9b, checkpoint model_400000.pth);
active on a token means ci > 0 after the lower-leaky-hard sigmoid, i.e.
pre-sigmoid > 0. Ours: C=8192 softpart components fit by IG-K=5 attribution
clustering on 1B streamed tokens; "active" is a ranking, so we take the top-k
by per-token attribution with k matched to THEIR L0 on that token -- both
sides then name the same number of active units.

The comparison is in WEIGHT SPACE, per token t and module m:

  M_ours(t, m) = sum_{c in top-L0(t)} share_c (.) W_m      [d_out, d_in]
  M_vpd(t, m)  = sum_{s: ci_s(t) > 0} U_s outer V_s        [d_out, d_in]

cosine(M_ours, M_vpd) over each module, plus pooled over all 24. Three nulls
isolate what the number means:
  rand_ours    random k of our components instead of the top-k (is it the
               attribution ranking, or just our weight coverage?)
  rand_vpd     random L0 of their subcomponents (same for theirs)
  shuffled     our M from token t vs their M from a different token t'
               (is any of it token-SPECIFIC?)

Also reported: component-level correspondence. For each of our active
components, the best-match cosine against every active VPD subcomponent in the
same module, computed without materialising either side:
  <share_c (.) W, u v^T> = u^T ((share_c (.) W) v)

    python3.12 compare_vpd.py --n_tokens 24
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sensor_study67 as S67
from sensor_study67 import MODULES, SENSORS, load67, capture

VPD_CKPT = Path("/dev/shm/vpd_decomp/model_400000.pth")
BANK = Path("/dev/shm/geo67_stream/bank_C8192_ig5.pt")
CI_HEADS, CI_DMODEL, CI_BLOCKS = 16, 2048, 8


def log(m):
    print(f"[compare] {m}", flush=True)


# ---------------- VPD causal-importance transformer, from their source ----

def rms(x):
    return F.rms_norm(x, (x.shape[-1],))


class CiBlock(nn.Module):
    """RMSNorm -> bidirectional RoPE attention -> residual -> RMSNorm -> MLP."""

    def __init__(self, sd, pre):
        super().__init__()
        self.q = sd[f"{pre}.attn.q_proj.weight"]
        self.k = sd[f"{pre}.attn.k_proj.weight"]
        self.v = sd[f"{pre}.attn.v_proj.weight"]
        self.o = sd[f"{pre}.attn.out_proj.weight"]
        self.inv = sd[f"{pre}.attn.rope.inv_freq"]
        self.w0, self.b0 = sd[f"{pre}.mlp.0.W"], sd[f"{pre}.mlp.0.b"]
        self.w2, self.b2 = sd[f"{pre}.mlp.2.W"], sd[f"{pre}.mlp.2.b"]

    def forward(self, x):
        h = rms(x)
        T = h.shape[-2]
        d = CI_DMODEL // CI_HEADS
        q = (h @ self.q.t()).view(*h.shape[:-1], CI_HEADS, d).transpose(-3, -2)
        k = (h @ self.k.t()).view(*h.shape[:-1], CI_HEADS, d).transpose(-3, -2)
        v = (h @ self.v.t()).view(*h.shape[:-1], CI_HEADS, d).transpose(-3, -2)
        t = torch.arange(T, device=x.device).float()
        f = torch.outer(t, self.inv)
        cos = torch.cat((f, f), -1).cos()[None, None]
        sin = torch.cat((f, f), -1).sin()[None, None]

        def rot(z):
            z1, z2 = z[..., : d // 2], z[..., d // 2:]
            return z * cos + torch.cat((-z2, z1), -1) * sin
        q, k = rot(q), rot(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        a = a.transpose(-3, -2).reshape(*h.shape)
        x = x + a @ self.o.t()
        h = rms(x)
        return x + (F.gelu(h @ self.w0 + self.b0) @ self.w2 + self.b2)


class CiFn(nn.Module):
    def __init__(self, sd, order, split):
        super().__init__()
        pre = "ci_fn._global_ci_fn"
        self.wi, self.bi = sd[f"{pre}._input_projector.W"], sd[f"{pre}._input_projector.b"]
        self.wo, self.bo = sd[f"{pre}._output_head.W"], sd[f"{pre}._output_head.b"]
        self.blocks = nn.ModuleList([CiBlock(sd, f"{pre}._blocks.{i}")
                                     for i in range(CI_BLOCKS)])
        self.order, self.split = order, split

    def forward(self, acts):
        x = torch.cat([rms(acts[n]) for n in self.order], -1)
        x = x @ self.wi + self.bi
        for b in self.blocks:
            x = b(x)
        out = x @ self.wo + self.bo
        return dict(zip(self.order, torch.split(out, self.split, -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n_tokens", type=int, default=24)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval_seqs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("out/compare_vpd.json"))
    args = ap.parse_args()
    dev = args.device
    torch.cuda.set_device(int(dev.split(":")[1]))
    t00 = time.perf_counter()

    # ---- their decomposition + CI net ----
    sd = torch.load(VPD_CKPT, map_location="cpu", weights_only=False)
    order = sorted(MODULES)                       # their sorted(layer names)
    key = {m: "_components." + m.replace(".", "-") for m in MODULES}
    U = {m: sd[key[m] + ".U"].float().to(dev) for m in MODULES}   # [C, d_out]
    V = {m: sd[key[m] + ".V"].float().to(dev) for m in MODULES}   # [d_in, C]
    split = [U[n].shape[0] for n in order]
    ci = CiFn({k: v.float().to(dev) for k, v in sd.items()
               if k.startswith("ci_fn")}, order, split).to(dev).eval()
    n_sub = sum(split)
    log(f"VPD: {n_sub} subcomponents across {len(order)} modules")

    # faithfulness of their U,V to the target (sanity; delta component covers
    # the残り, so this need not be tiny -- report it, don't assert on it)
    tgt = load67(dev, "plain")
    W0 = {m: tgt.get_submodule(m).weight.detach().clone() for m in MODULES}
    rel = {m: float((V[m] @ U[m]).t().sub(W0[m]).norm() / W0[m].norm())
           for m in MODULES}
    log(f"||UV - W||/||W||: min {min(rel.values()):.3f} "
        f"max {max(rel.values()):.3f} (delta component absorbs this)")

    # ---- our decomposition ----
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    C = bank["C"]
    sidx = {m: bank["sidx"][m].to(dev) for m in MODULES}
    swgt = {m: bank["swgt"][m].to(dev) for m in MODULES}
    log(f"ours: C={C} softpart bank from {bank['tokens']/1e6:.0f}M tokens, "
        f"sensor {bank['sensor']}")

    cfg = S67.SENSORS[bank["sensor"]]
    model = load67(dev, cfg["mode"])
    Wt = {m: model.get_submodule(m).weight for m in MODULES}
    K = cfg["K"]

    # ---- shared eval tokens (stratified) ----
    from transformers import AutoTokenizer
    from pile_data import load_pile_blocks
    tok = AutoTokenizer.from_pretrained(S67.TOKENIZER)
    ids_cpu, _, _ = load_pile_blocks(tok, 168, args.seq, seed=0,
                                     tokenizer_name=S67.TOKENIZER)
    eval_ids = ids_cpu[-args.eval_seqs:].to(dev)
    gsel = torch.Generator().manual_seed(12345)
    samp = [(int(torch.randint(0, eval_ids.shape[0], (1,), generator=gsel)),
             int(torch.randint(64, args.seq - 2, (1,), generator=gsel)))
            for _ in range(args.n_tokens)]

    # ---- per-token machinery ----
    def our_attr(b, t):
        """Per-token attribution of every one of our components (IG path)."""
        A = {m: torch.zeros_like(W0[m]) for m in MODULES}
        for step in range(K):
            a = (step + 1) / K
            with torch.no_grad():
                for m in MODULES:
                    Wt[m].copy_(W0[m] * a)
            pre, post, hs = {}, {}, []
            for m in MODULES:
                mod = model.get_submodule(m)

                def hook(mm, inp, out, _m=m):
                    pre[_m] = inp[0]
                    out.retain_grad()
                    post[_m] = out
                hs.append(mod.register_forward_hook(hook))
            lg = model(eval_ids[b:b + 1])
            for h in hs:
                h.remove()
            reward = lg[0, t, eval_ids[b, t + 1]].float()
            grads = torch.autograd.grad(reward, [post[m] for m in MODULES])
            for m, g in zip(MODULES, grads):
                A[m] += (g[0].float().t() @ pre[m][0].float()) / K
            del pre, post, grads, lg
        with torch.no_grad():
            for m in MODULES:
                Wt[m].copy_(W0[m])
        v = torch.zeros(C, device=dev, dtype=torch.float64)
        with torch.no_grad():
            for m in MODULES:
                v += torch.bincount(
                    sidx[m].reshape(-1).int(),
                    weights=(swgt[m].float() * (A[m] * W0[m])[None]
                             ).reshape(-1).double(), minlength=C)
        return v

    @torch.no_grad()
    def vpd_active(b, t):
        """Their active subcomponents at position t: ci pre-sigmoid > 0."""
        pre, hs = {}, []
        for m in MODULES:
            mod = tgt.get_submodule(m)

            def hook(mm, inp, out, _m=m):
                pre[_m] = inp[0]
            hs.append(mod.register_forward_hook(hook))
        tgt(eval_ids[b:b + 1])
        for h in hs:
            h.remove()
        logits = ci({m: pre[m] for m in MODULES})
        return {m: (logits[m][0, t] > 0) for m in MODULES}

    @torch.no_grad()
    def our_mask(active_set):
        """sum of share_c (.) W over the active components, per module."""
        out = {}
        act = torch.zeros(C, dtype=torch.bool, device=dev)
        act[active_set] = True
        for m in MODULES:
            sel = act[sidx[m].int()]
            out[m] = (swgt[m].float() * sel).sum(0) * W0[m]
        return out

    @torch.no_grad()
    def vpd_mask(active):
        return {m: (V[m][:, active[m]] @ U[m][active[m]]).t()
                for m in MODULES}

    def cos(a, b):
        num = sum(float((a[m] * b[m]).sum()) for m in MODULES)
        na = math.sqrt(sum(float((a[m] ** 2).sum()) for m in MODULES))
        nb = math.sqrt(sum(float((b[m] ** 2).sum()) for m in MODULES))
        return num / max(na * nb, 1e-30)

    def cos_per_module(a, b):
        out = {}
        for m in MODULES:
            n = float((a[m] * b[m]).sum())
            d = float(a[m].norm()) * float(b[m].norm())
            out[m] = n / max(d, 1e-30)
        return out

    # ---- run ----
    rng = torch.Generator(device=dev).manual_seed(7)
    rows = []
    prev_vpd = None
    for i, (b, t) in enumerate(samp):
        act_v = vpd_active(b, t)
        L0 = int(sum(int(v.sum()) for v in act_v.values()))
        attr = our_attr(b, t)
        top = torch.argsort(attr, descending=True)[:L0]
        Mo = our_mask(top)
        Mv = vpd_mask(act_v)
        # nulls
        rnd_ours = our_mask(torch.randperm(C, generator=rng, device=dev)[:L0])
        rand_v = {}
        for m in MODULES:
            k = int(act_v[m].sum())
            pick = torch.randperm(act_v[m].numel(), generator=rng,
                                  device=dev)[:k]
            z = torch.zeros_like(act_v[m])
            z[pick] = True
            rand_v[m] = z
        Mrv = vpd_mask(rand_v)
        row = {"b": b, "t": t, "L0_vpd": L0,
               "cos": cos(Mo, Mv),
               "cos_rand_ours": cos(rnd_ours, Mv),
               "cos_rand_vpd": cos(Mo, Mrv),
               "cos_shuffled": cos(Mo, prev_vpd) if prev_vpd else None,
               "per_module": cos_per_module(Mo, Mv)}
        # component-level best match: u^T ((share_c . W) v), our top-32 comps
        best = []
        with torch.no_grad():
            for c in top[:32].tolist():
                num_best, no = 0.0, 0.0
                for m in MODULES:
                    selc = (sidx[m].int() == c)
                    if not selc.any():
                        continue
                    Wc = (swgt[m].float() * selc).sum(0) * W0[m]
                    no += float((Wc ** 2).sum())
                    aidx = torch.nonzero(act_v[m], as_tuple=True)[0]
                    if aidx.numel() == 0:
                        continue
                    Us, Vs = U[m][aidx], V[m][:, aidx]
                    num = (Us * (Wc @ Vs).t()).sum(1)          # [n_active]
                    den = Us.norm(dim=1) * Vs.norm(dim=0)
                    val = (num / den.clamp_min(1e-30))
                    num_best = max(num_best, float(val.abs().max()))
                best.append(num_best / max(math.sqrt(no), 1e-30))
        row["best_match_mean"] = float(np.mean(best))
        rows.append(row)
        prev_vpd = Mv
        log(f"tok {i:>2} (seq {b}, pos {t})  L0={L0:>4}  cos={row['cos']:.3f}  "
            f"rand_ours={row['cos_rand_ours']:.3f}  "
            f"rand_vpd={row['cos_rand_vpd']:.3f}  "
            f"best_match={row['best_match_mean']:.3f}")

    agg = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
           for k in ("cos", "cos_rand_ours", "cos_rand_vpd", "cos_shuffled",
                     "best_match_mean")}
    agg["L0_mean"] = float(np.mean([r["L0_vpd"] for r in rows]))
    pm = {m: float(np.mean([r["per_module"][m] for r in rows]))
          for m in MODULES}
    out = {"format": "compare_vpd_v1", "ours": {"C": C,
           "sensor": bank["sensor"], "tokens": int(bank["tokens"])},
           "theirs": {"run": "goodfire/spd/s-55ea3f9b", "n_sub": n_sub,
                      "ckpt": "model_400000.pth"},
           "uv_vs_target_relerr": rel, "n_tokens": len(rows),
           "aggregate": agg, "per_module_cos": pm, "per_token": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log(f"AGG cos={agg['cos']:.3f}  rand_ours={agg['cos_rand_ours']:.3f}  "
        f"rand_vpd={agg['cos_rand_vpd']:.3f}  shuffled={agg['cos_shuffled']:.3f}")
    log(f"wrote {args.out} ({time.perf_counter()-t00:.0f}s)")


if __name__ == "__main__":
    main()
