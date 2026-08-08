"""Cross-layer virtual attention heads inside a single component.

A virtual head is not a head. It is a PAIR of heads in different layers whose
circuits multiply through the residual stream: head A writes a subspace, head B
reads it. Elhage et al. score that with

    comp(A -> B) = ||W_B W_OV^A||_F / (||W_B||_F ||W_OV^A||_F)

for W_B in {W_Q, W_K, W_V} — Q-, K- and V-composition. K-composition is the
induction mechanism: a previous-token head writes "the token before me", and a
later head's KEY side reads it.

The question this asks of the decomposition is sharper than "is a component
interpretable". Components were fit on attribution geometry, with no notion of
layers, heads, or composition anywhere in the objective. If a component's
attention mass nonetheless lands on head pairs that compose ABOVE the rate of
mass-matched random pairs, then the decomposition is recovering cross-layer
circuits it was never told about.

Two distinct claims are measured separately, because they can come apart:

  grouping     do heads that SHARE a component compose more than chance pairs?
               (full-weight composition; about the partition)
  internal     does the component's OWN SLICE of those heads compose?
               (masked-weight composition; about the component as a circuit)

    python3.12 virtual_heads.py --components 3392 108 3634
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401
from german_vpd_1b import log

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj")


def head_slices(target, L, H, KV, HD):
    """Per-head weight blocks. Under GQA query head h reads kv head h // (H/KV)."""
    W = {k: [] for k in KINDS}
    for l in range(L):
        p = f"hf.model.layers.{l}"
        for k in KINDS:
            W[k].append(target.get_submodule(f"{p}.self_attn.{k}")
                        .weight.detach().float())
    return W


def composition(WB, WO, GV, eps=1e-30):
    """comp = ||WB @ WO @ WV||_F / (||WB||_F * ||WO @ WV||_F), batched.

    WB  [b2, HD, D]   the reading head's Q/K/V projection
    WO  [b1, D, HD]   the writing head's output projection
    GV  [b1, HD, HD]  W_V W_V^T for the writing head, so W_V is never
                      materialised into a D x D product
    Returns [b2, b1].
    """
    # M[b2,b1] = WB[b2] @ WO[b1]  ->  [b2, b1, HD, HD]
    M = torch.einsum("aij,bjk->abik", WB, WO)
    MtM = torch.einsum("abik,abil->abkl", M, M)
    num = torch.einsum("abkl,bkl->ab", MtM, GV).clamp_min(0).sqrt()
    # ||W_OV||_F^2 = <WO^T WO, GV>
    OtO = torch.einsum("bji,bjk->bik", WO, WO)
    ov = torch.einsum("bik,bik->b", OtO, GV).clamp_min(0).sqrt()
    nb = WB.reshape(WB.shape[0], -1).norm(dim=1)
    return num / (nb[:, None] * ov[None, :] + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--components", type=int, nargs="+",
                    default=[3392, 108, 3634, 3203])
    ap.add_argument("--top_heads", type=int, default=6)
    ap.add_argument("--null_samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="virtual_heads.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    target = geo1b.load_target_1b(dev)
    cfg = target.hf.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    KV, HD = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    D, REP = cfg.hidden_size, H // cfg.num_key_value_heads
    W = head_slices(target, L, H, KV, HD)
    log(f"{L} layers x {H} heads (kv {KV}, head_dim {HD})")

    # ---- full-weight per-head blocks ----
    WQ = torch.stack([W["q_proj"][l][h * HD:(h + 1) * HD]
                      for l in range(L) for h in range(H)])          # [LH,HD,D]
    WK = torch.stack([W["k_proj"][l][(h // REP) * HD:(h // REP + 1) * HD]
                      for l in range(L) for h in range(H)])
    WV = torch.stack([W["v_proj"][l][(h // REP) * HD:(h // REP + 1) * HD]
                      for l in range(L) for h in range(H)])
    WO = torch.stack([W["o_proj"][l][:, h * HD:(h + 1) * HD]
                      for l in range(L) for h in range(H)])          # [LH,D,HD]
    GV = torch.einsum("bij,bkj->bik", WV, WV)                        # [LH,HD,HD]

    def all_pairs(WB):
        """comp[reader, writer] for every ordered head pair, masked to L1<L2."""
        out = torch.zeros(L * H, L * H, device=dev)
        for l2 in range(1, L):
            r = slice(l2 * H, (l2 + 1) * H)
            out[r, :l2 * H] = composition(WB[r], WO[:l2 * H], GV[:l2 * H])
        return out

    log("composition scores over all 512x512 head pairs")
    comp = {"Q": all_pairs(WQ), "K": all_pairs(WK), "V": all_pairs(WV)}
    causal = torch.zeros(L * H, L * H, dtype=torch.bool, device=dev)
    for l2 in range(1, L):
        causal[l2 * H:(l2 + 1) * H, :l2 * H] = True

    # ---- per-(component, layer, head) owned attention mass ----
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    C = int(bank["C"])
    mass = torch.zeros(C, L * H, device=dev)
    for path in bank["modules"]:
        kind = path.rsplit(".", 1)[1]
        if kind not in ("q_proj", "o_proj"):
            continue          # q rows / o columns resolve per QUERY head exactly
        l = int(path.split("layers.")[1].split(".")[0])
        w2 = target.get_submodule(path).weight.detach().float() ** 2
        sidx, swgt = bank["sidx"][path].to(dev), bank["swgt"][path].to(dev)
        for h in range(H):
            if kind == "q_proj":
                sl = (slice(None), slice(h * HD, (h + 1) * HD), slice(None))
                wsl = w2[h * HD:(h + 1) * HD]
            else:
                sl = (slice(None), slice(None), slice(h * HD, (h + 1) * HD))
                wsl = w2[:, h * HD:(h + 1) * HD]
            contrib = (swgt[sl].float() * wsl).reshape(swgt.shape[0], -1)
            mass[:, l * H + h] += torch.bincount(
                sidx[sl].reshape(-1).long(), weights=contrib.reshape(-1),
                minlength=C).float()
        del w2, sidx, swgt
    del bank
    total = mass.sum(1).clamp_min(1e-30)
    share = mass / total[:, None]
    log("per-(component, head) attention mass done")

    # ---- how many layers does a component's attention mass span? ----
    per_layer = share.view(C, L, H).sum(2)
    span = {f"ge_{int(t * 100)}pct": float((per_layer >= t).sum(1).float().mean())
            for t in (0.01, 0.05, 0.10)}
    top_layer_share = per_layer.max(1).values
    log(f"mean layers holding >=5% of a component's attention mass: "
        f"{span['ge_5pct']:.2f}; mean top-layer share "
        f"{float(top_layer_share.mean()):.3f}")

    # ---- grouping test: do co-owned head pairs compose above chance? ----
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    results = {}
    for ctype, M in comp.items():
        Mc = M[causal]
        obs, weights = [], []
        for c in range(C):
            s = share[c]
            top = torch.topk(s, args.top_heads)
            hs, sv = top.indices, top.values
            if float(sv[1]) < 0.02:
                continue                     # needs at least two real heads
            for i in range(args.top_heads):
                for j in range(args.top_heads):
                    a, b = int(hs[i]), int(hs[j])
                    if not causal[a, b]:
                        continue
                    obs.append(float(M[a, b]))
                    weights.append(float(sv[i] * sv[j]))
        obs_t = torch.tensor(obs)
        w_t = torch.tensor(weights)
        null = Mc[torch.randint(0, Mc.numel(), (args.null_samples,),
                                generator=g).to(Mc.device)].cpu()
        results[ctype] = {
            "n_pairs": len(obs),
            "co_owned_mean": float(obs_t.mean()),
            "co_owned_mass_weighted": float((obs_t * w_t).sum() / w_t.sum()),
            "null_mean": float(null.mean()),
            "null_p95": float(null.quantile(0.95)),
            "ratio": float(obs_t.mean() / null.mean()),
            "frac_above_null_p95": float((obs_t > null.quantile(0.95))
                                         .float().mean()),
        }
        r = results[ctype]
        log(f"{ctype}-composition: co-owned {r['co_owned_mean']:.4f} vs null "
            f"{r['null_mean']:.4f}  ratio {r['ratio']:.2f}x  "
            f"{100 * r['frac_above_null_p95']:.1f}% above null p95  "
            f"(n={r['n_pairs']})")

    # ---- named components: the actual virtual heads ----
    detail = {}
    for c in args.components:
        s = share[c]
        top = torch.topk(s, args.top_heads)
        heads = [{"layer": int(i) // H, "head": int(i) % H,
                  "mass_share": round(float(v), 4)}
                 for i, v in zip(top.indices, top.values)]
        pairs = []
        for i in range(args.top_heads):
            for j in range(args.top_heads):
                a, b = int(top.indices[i]), int(top.indices[j])
                if not causal[a, b]:
                    continue
                pairs.append({
                    "writer": f"L{b // H}H{b % H}", "reader": f"L{a // H}H{a % H}",
                    "writer_share": round(float(top.values[j]), 4),
                    "reader_share": round(float(top.values[i]), 4),
                    **{f"{k}_comp": round(float(comp[k][a, b]), 5)
                       for k in ("Q", "K", "V")},
                })
        pairs.sort(key=lambda p: -p["K_comp"])
        detail[str(c)] = {"top_heads": heads, "cross_layer_pairs": pairs[:12],
                          "layers_spanned_5pct":
                              int((per_layer[c] >= 0.05).sum())}
        log(f"c{c}: heads " + ", ".join(
            f"L{h['layer']}H{h['head']}({100 * h['mass_share']:.0f}%)"
            for h in heads[:4]))
        for p in pairs[:3]:
            log(f"    {p['writer']} -> {p['reader']}  "
                f"K {p['K_comp']:.4f}  Q {p['Q_comp']:.4f}  V {p['V_comp']:.4f}")

    out = {"format": "virtual_heads_v1", "L": L, "H": H, "C": C,
           "layer_span": span,
           "mean_top_layer_share": float(top_layer_share.mean()),
           "grouping_test": results, "components": detail,
           "null_percentiles": {k: {q: float(comp[k][causal].quantile(q))
                                    for q in (0.5, 0.9, 0.95, 0.99)}
                                for k in comp}}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
