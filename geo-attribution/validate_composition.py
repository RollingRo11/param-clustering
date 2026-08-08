"""Does the Frobenius composition score detect a circuit we KNOW is there?

virtual_heads.py finds no elevated composition among co-owned head pairs. That
is only evidence about the decomposition if the score can see composition at
all. So this measures it on the one cross-layer circuit in this model that is
independently established: previous-token head -> induction head, the textbook
case of K-composition.

Both head sets are found behaviourally, from attention patterns, with no
reference to weights:

  previous-token score   mean attention from position i to i-1 on random text
  induction score        attention from the second copy of a token to the token
                         that followed its first occurrence

If K-composition between those two sets is elevated over the null of random
causal head pairs, the score works and the null in virtual_heads.py is a real
statement about the partition. If it is flat, the score is too weak at this
scale and the null means nothing.

    python3.12 validate_composition.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401
from german_vpd_1b import log
from induction4096 import repeated_batch
from virtual_heads import composition


@torch.no_grad()
def attention_maps(hf, idx, L, H):
    """[L, B, H, T, T] attention probabilities.

    sdpa never materialises the probability matrix, so the flag has to be
    flipped on the config and put back afterwards.
    """
    hf.config._attn_implementation = "eager"
    try:
        out = hf(idx, output_attentions=True)
        return torch.stack(list(out.attentions))
    finally:
        hf.config._attn_implementation = "sdpa"


@torch.no_grad()
def behavioural_scores(hf, cfg, dev, span, n_seq, warmup):
    """(previous-token score, induction score) per head, both [L, H]."""
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    S = span
    idx = repeated_batch(n_seq, S, 1000, 20000, cfg.bos_token_id, dev, 0)
    A = attention_maps(hf, idx, L, H)                     # [L,B,H,T,T]
    T = idx.shape[1]

    prev = torch.zeros(L, H, device=dev)
    ind = torch.zeros(L, H, device=dev)
    q = torch.arange(warmup + 1, T, device=dev)
    prev_k = q - 1
    prev[:, :] = A[:, :, :, q, prev_k].mean(dim=(1, 3))

    # induction: from the second copy at position p, look at p - S + 1, which
    # is the token that FOLLOWED the first occurrence of the token at p
    q2 = torch.arange(S + 1 + warmup, 2 * S, device=dev)
    k2 = q2 - S + 1
    ind[:, :] = A[:, :, :, q2, k2].mean(dim=(1, 3))
    return prev.cpu().numpy(), ind.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--span", type=int, default=48)
    ap.add_argument("--n_seq", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--null_samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", type=Path,
                    default=geo1b.SHM_ROOT / "run1b_streamC4096")
    ap.add_argument("--out", default="validate_composition.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))

    target = geo1b.load_target_1b(dev)
    hf, cfg = target.hf, target.hf.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    HD, REP = cfg.hidden_size // H, H // cfg.num_key_value_heads

    prev, ind = behavioural_scores(hf, cfg, dev, args.span, args.n_seq,
                                   args.warmup)
    pv = [(int(i // H), int(i % H), float(prev.reshape(-1)[i]))
          for i in np.argsort(-prev.reshape(-1))[:args.top]]
    iv = [(int(i // H), int(i % H), float(ind.reshape(-1)[i]))
          for i in np.argsort(-ind.reshape(-1))[:args.top]]
    log("top previous-token heads: " +
        ", ".join(f"L{l}H{h}({s:.2f})" for l, h, s in pv))
    log("top induction heads:      " +
        ", ".join(f"L{l}H{h}({s:.2f})" for l, h, s in iv))

    # ---- composition over all head pairs ----
    W = {k: [target.get_submodule(f"hf.model.layers.{l}.self_attn.{k}")
             .weight.detach().float() for l in range(L)]
         for k in ("q_proj", "k_proj", "v_proj", "o_proj")}
    WK = torch.stack([W["k_proj"][l][(h // REP) * HD:(h // REP + 1) * HD]
                      for l in range(L) for h in range(H)])
    WQ = torch.stack([W["q_proj"][l][h * HD:(h + 1) * HD]
                      for l in range(L) for h in range(H)])
    WV = torch.stack([W["v_proj"][l][(h // REP) * HD:(h // REP + 1) * HD]
                      for l in range(L) for h in range(H)])
    WO = torch.stack([W["o_proj"][l][:, h * HD:(h + 1) * HD]
                      for l in range(L) for h in range(H)])
    GV = torch.einsum("bij,bkj->bik", WV, WV)

    comp = {}
    for name, WB in (("Q", WQ), ("K", WK), ("V", WV)):
        M = torch.zeros(L * H, L * H, device=dev)
        for l2 in range(1, L):
            r = slice(l2 * H, (l2 + 1) * H)
            M[r, :l2 * H] = composition(WB[r], WO[:l2 * H], GV[:l2 * H])
        comp[name] = M
    causal = torch.zeros(L * H, L * H, dtype=torch.bool, device=dev)
    for l2 in range(1, L):
        causal[l2 * H:(l2 + 1) * H, :l2 * H] = True

    g = torch.Generator().manual_seed(args.seed)
    out = {"top_prev_heads": pv, "top_induction_heads": iv, "tests": {}}
    for name, M in comp.items():
        Mc = M[causal].cpu()
        null = Mc[torch.randint(0, Mc.numel(), (args.null_samples,),
                                generator=g)]
        vals = []
        for pl, ph, _ in pv:
            for il, ih, _ in iv:
                a, b = il * H + ih, pl * H + ph      # reader, writer
                if causal[a, b]:
                    vals.append(float(M[a, b]))
        if not vals:
            continue
        v = torch.tensor(vals)
        out["tests"][name] = {
            "n_pairs": len(vals),
            "prev_to_induction_mean": round(float(v.mean()), 5),
            "prev_to_induction_max": round(float(v.max()), 5),
            "null_mean": round(float(null.mean()), 5),
            "null_p95": round(float(null.quantile(0.95)), 5),
            "null_p99": round(float(null.quantile(0.99)), 5),
            "ratio": round(float(v.mean() / null.mean()), 3),
            "frac_above_null_p95": round(float((v > null.quantile(0.95))
                                               .float().mean()), 3),
        }
        r = out["tests"][name]
        log(f"{name}-composition  prev->induction {r['prev_to_induction_mean']:.4f} "
            f"(max {r['prev_to_induction_max']:.4f}) vs null {r['null_mean']:.4f} "
            f"| ratio {r['ratio']:.2f}x | {100 * r['frac_above_null_p95']:.0f}% "
            f"above null p95  (n={r['n_pairs']})")

    (args.run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"wrote {args.run_dir / args.out}")


if __name__ == "__main__":
    main()
