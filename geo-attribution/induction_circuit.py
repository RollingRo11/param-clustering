"""Locate the model's induction circuit independently, then ask whether c108
sits on it.

Three independent measurements, none of which uses the decomposition:

  heads   - per-head induction score in the Olsson sense: on [BOS, R, R], how
            much attention flows from a copy-2 position to the token that
            FOLLOWED the same token in copy 1. Pure attention pattern.
  ablate  - zero each layer's attention block / MLP block in turn and measure
            what happens to copy-2 loss. Pure causal, no attribution.
  head-ko - zero individual heads (their slice of the o_proj input).

Then `compare` puts c108's owned squared weight mass next to all of it. The
attention matrices are head-sliced (q rows, o columns), so component ownership
can be resolved per head and compared to the induction scores directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401
from german_vpd_1b import log
from induction4096 import repeated_batch, control_batch, ce_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--component", type=int, default=108)
    parser.add_argument("--span", type=int, default=64)
    parser.add_argument("--n_seq", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--top_heads", type=int, default=12)
    args = parser.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    target = geo1b.load_target_1b(dev)
    hf = target.hf
    cfg = hf.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    HD = cfg.hidden_size // H
    S = args.span
    bos = cfg.bos_token_id
    idx = repeated_batch(args.n_seq, S, 1000, 20000, bos, dev, 0)
    first = torch.zeros_like(idx, dtype=torch.bool)
    second = torch.zeros_like(idx, dtype=torch.bool)
    first[:, 1 + args.warmup:S] = True
    second[:, S + 1 + args.warmup:2 * S] = True
    ctrl = control_batch(tok, 96, dev)
    cmask = torch.ones_like(ctrl, dtype=torch.bool)
    cmask[:, :4] = False

    def evaluate():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=True):
            lg = hf(idx).logits
            d2 = ce_at(lg, idx, second)
            del lg
            lc = hf(ctrl).logits
            cc = ce_at(lc, ctrl, cmask)
            del lc
        return d2, cc

    b2, bc = evaluate()
    log(f"base copy2 CE {b2:.4f}   control CE {bc:.4f}")

    # ---- 1. induction scores from attention patterns ----
    cfg._attn_implementation = "eager"
    with torch.no_grad():
        out = hf(idx, output_attentions=True)
    pos = torch.arange(S + 1 + args.warmup, 2 * S, device=dev)
    tgt = pos - S + 1                      # token that followed it in copy 1
    scores = np.zeros((L, H))
    for l, att in enumerate(out.attentions):        # [B, H, T, T]
        sel = att[:, :, pos, :][:, :, torch.arange(len(pos)), tgt]
        scores[l] = sel.float().mean(dim=(0, 2)).cpu().numpy() \
            if sel.dim() == 3 else sel.float().mean(0).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    cfg._attn_implementation = "sdpa"
    flat = np.dstack(np.unravel_index(np.argsort(-scores, axis=None),
                                      scores.shape))[0]
    log(f"top {args.top_heads} induction heads (attention to the "
        f"copy-1 successor):")
    top = []
    for l, h in flat[:args.top_heads]:
        top.append((int(l), int(h), float(scores[l, h])))
        log(f"  L{l:<2} H{h:<2}  induction score {scores[l, h]:.3f}")

    # ---- 2. block ablations ----
    def run_with(hook_mod, fn):
        handle = hook_mod.register_forward_hook(fn)
        try:
            return evaluate()
        finally:
            handle.remove()

    zero = lambda m, i, o: torch.zeros_like(o[0] if isinstance(o, tuple) else o) \
        if not isinstance(o, tuple) else (torch.zeros_like(o[0]),) + o[1:]
    blocks = {}
    for l in range(L):
        for name, mod in (("attn", hf.model.layers[l].self_attn),
                          ("mlp", hf.model.layers[l].mlp)):
            d2, cc = run_with(mod, zero)
            blocks[f"L{l}.{name}"] = {"copy2_ce": d2, "d_induction": d2 - b2,
                                      "control_ce": cc, "d_control": cc - bc}
    log("block ablation — largest induction damage:")
    for k, v in sorted(blocks.items(), key=lambda x: -x[1]["d_induction"])[:8]:
        log(f"  {k:<10} induction {v['d_induction']:+8.3f}  "
            f"control {v['d_control']:+7.3f}")

    # ---- 3. individual head knockouts ----
    head_ko = {}
    for l, h, _ in top:
        o_proj = hf.model.layers[l].self_attn.o_proj

        def pre(mod, inputs, _h=h):
            x = inputs[0].clone()
            x[..., _h * HD:(_h + 1) * HD] = 0
            return (x,) + inputs[1:]

        handle = o_proj.register_forward_pre_hook(pre)
        try:
            d2, cc = evaluate()
        finally:
            handle.remove()
        head_ko[f"L{l}H{h}"] = {"d_induction": d2 - b2, "d_control": cc - bc}
    log("individual head knockouts:")
    for k, v in sorted(head_ko.items(), key=lambda x: -x[1]["d_induction"]):
        log(f"  {k:<8} induction {v['d_induction']:+8.3f}  "
            f"control {v['d_control']:+7.3f}")

    # ---- 4. c108's owned weight mass, per matrix and per head ----
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    c = args.component
    per_matrix, per_head = {}, np.zeros((L, H))
    for path in bank["modules"]:
        w2 = target.get_submodule(path).weight.detach().float().cpu() ** 2
        sidx, swgt = bank["sidx"][path], bank["swgt"][path]
        owned = torch.zeros_like(w2)
        for s in range(sidx.shape[0]):
            owned += (sidx[s] == c).float() * swgt[s].float() * w2
        per_matrix[path] = float(owned.sum())
        l = int(path.split("layers.")[1].split(".")[0])
        kind = path.rsplit(".", 1)[1]
        if kind == "q_proj":                      # head h owns rows h*HD..
            for h in range(H):
                per_head[l, h] += float(owned[h * HD:(h + 1) * HD].sum())
        elif kind == "o_proj":                    # head h owns columns
            for h in range(H):
                per_head[l, h] += float(owned[:, h * HD:(h + 1) * HD].sum())
    total = sum(per_matrix.values())
    attn_mass = sum(v for k, v in per_matrix.items()
                    if k.endswith(("q_proj", "k_proj", "v_proj", "o_proj")))
    log(f"c{c}: total owned squared weight mass {total:.4g}")
    log(f"  attention share {100*attn_mass/total:.1f}%   "
        f"MLP share {100*(1-attn_mass/total):.1f}%")

    # ---- 5. does the mass sit on the induction heads? ----
    ind_flat = scores.reshape(-1)
    mass_flat = per_head.reshape(-1)
    order = np.argsort(-ind_flat)
    k = args.top_heads
    top_mass = mass_flat[order[:k]].sum() / max(mass_flat.sum(), 1e-30)
    log(f"  share of c{c}'s ATTENTION mass sitting on the top-{k} induction "
        f"heads: {100*top_mass:.1f}%  (chance {100*k/(L*H):.1f}%)")
    rho = np.corrcoef(ind_flat, mass_flat)[0, 1]
    log(f"  corr(head induction score, c{c} head mass) = {rho:+.3f}")

    out_path = run_dir / f"induction_circuit_c{c}.json"
    out_path.write_text(json.dumps(
        {"base": {"copy2_ce": b2, "control_ce": bc},
         "head_scores": scores.tolist(), "top_heads": top,
         "blocks": blocks, "head_ko": head_ko,
         "component": c, "per_matrix": per_matrix,
         "per_head_mass": per_head.tolist(),
         "attn_share": attn_mass / total}, indent=1))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
