"""Which component, if any, owns the ground-truth induction heads?

c108 damages induction behaviour but leaves the attention pattern intact, so it
is the readout, not the circuit. This searches all C components for the one
whose owned weight mass actually sits on the induction heads.

q_proj rows and o_proj columns are sliced per query head, so ownership resolves
per head exactly. (k/v are shared by 4 query heads under GQA and are reported
separately rather than mixed into the per-head score.)

Score = the induction-score-weighted average over a component's attention mass:
high only when the mass concentrates on heads that actually do induction.

The decisive check is not the loss but the ATTENTION PATTERN: a component that
truly holds the circuit should degrade the heads' induction score when scaled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401
from german_vpd_1b import log
from induction4096 import repeated_batch, ce_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--span", type=int, default=64)
    parser.add_argument("--n_seq", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--show", type=int, default=15)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[8.0, 16.0, 32.0])
    args = parser.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    target = geo1b.load_target_1b(dev)
    hf = target.hf
    cfg = hf.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    HD = cfg.hidden_size // H
    S = args.span
    idx = repeated_batch(args.n_seq, S, 1000, 20000, cfg.bos_token_id, dev, 0)
    second = torch.zeros_like(idx, dtype=torch.bool)
    second[:, S + 1 + args.warmup:2 * S] = True
    pos = torch.arange(S + 1 + args.warmup, 2 * S, device=dev)
    tgt = pos - S + 1

    def head_scores():
        cfg._attn_implementation = "eager"
        with torch.no_grad():
            out = hf(idx, output_attentions=True)
        sc = np.zeros((L, H))
        for l, att in enumerate(out.attentions):
            sel = att[:, :, pos, :][:, :, torch.arange(len(pos)), tgt]
            sc[l] = (sel.float().mean(dim=(0, 2)).cpu().numpy()
                     if sel.dim() == 3 else sel.float().mean(0).cpu().numpy())
        del out
        torch.cuda.empty_cache()
        cfg._attn_implementation = "sdpa"
        return sc

    def copy2_ce():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=True):
            return ce_at(hf(idx).logits, idx, second)

    base_sc = head_scores()
    base_ce = copy2_ce()
    log(f"base copy-2 CE {base_ce:.3f}; strongest head "
        f"L{base_sc.argmax() // H}H{base_sc.argmax() % H} "
        f"score {base_sc.max():.3f}")

    # ---- per-(component, head) owned weight mass over all C ----
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    C = int(bank["C"])
    mass = torch.zeros(C, L, H, device=dev)
    for path in bank["modules"]:
        kind = path.rsplit(".", 1)[1]
        if kind not in ("q_proj", "o_proj"):
            continue
        l = int(path.split("layers.")[1].split(".")[0])
        w2 = target.get_submodule(path).weight.detach().float().to(dev) ** 2
        sidx = bank["sidx"][path].to(dev)
        swgt = bank["swgt"][path].to(dev)
        for h in range(H):
            sl = (slice(None), slice(h * HD, (h + 1) * HD), slice(None)) \
                if kind == "q_proj" \
                else (slice(None), slice(None), slice(h * HD, (h + 1) * HD))
            wsl = w2[h * HD:(h + 1) * HD] if kind == "q_proj" \
                else w2[:, h * HD:(h + 1) * HD]
            contrib = (swgt[sl].float() * wsl).reshape(swgt.shape[0], -1)
            mass[:, l, h] += torch.bincount(
                sidx[sl].reshape(-1).long(), weights=contrib.reshape(-1),
                minlength=C).float()
        del w2, sidx, swgt
        log(f"  per-head mass: {path}")
    mass_c = mass.cpu().numpy()
    total = mass_c.reshape(C, -1).sum(1)
    sc_flat = base_sc.reshape(-1)
    weighted = (mass_c.reshape(C, -1) * sc_flat).sum(1) / np.maximum(total, 1e-30)
    top_head = np.unravel_index(base_sc.argmax(), base_sc.shape)
    share_top = mass_c[:, top_head[0], top_head[1]] / np.maximum(total, 1e-30)

    catalog = {}
    cat_path = Path(__file__).parent / "out/catalog_prop1b_C4096.json"
    if cat_path.exists():
        catalog = json.loads(cat_path.read_text())
    order = np.argsort(-weighted)
    log(f"mean induction-weighted score over all components: {weighted.mean():.4f}")
    log(f"top {args.show} components by induction-head-weighted attention mass:")
    for c in order[:args.show]:
        log(f"  c{c:<5} score {weighted[c]:.3f}  "
            f"share on L{top_head[0]}H{top_head[1]} {100*share_top[c]:5.1f}%  "
            f"{catalog.get(str(c), {}).get('label', '')[:52]}")

    # ---- causal + attention-pattern test on the leaders ----
    from german_permatrix import PerMatrixEditor
    cands = [int(c) for c in order[:args.test]]
    log(f"causal test on {cands} — does the attention PATTERN move?")
    results = []
    top12 = np.argsort(-sc_flat)[:12]
    for c in cands:
        ed = PerMatrixEditor(target, bank, [c], dev)
        nm = len(ed.modules)
        for a in args.alphas:
            saved = ed.apply_in_place(torch.full((1, nm), a, device=dev))
            ed.alpha = None
            sc2 = head_scores()
            ce2 = copy2_ce()
            ed.restore(saved)
            drop = 100 * (sc2.reshape(-1)[top12].mean()
                          / sc_flat[top12].mean() - 1)
            results.append({"component": c, "alpha": a, "copy2_ce": ce2,
                            "d_induction": ce2 - base_ce,
                            "top_head_score": float(sc2[top_head]),
                            "top12_pattern_change_pct": float(drop)})
            log(f"  c{c} a={a:g}: copy2 CE {ce2:7.3f} (+{ce2-base_ce:6.3f})  "
                f"L{top_head[0]}H{top_head[1]} {base_sc[top_head]:.3f}->"
                f"{sc2[top_head]:.3f}  top-12 pattern {drop:+.1f}%")
        del ed
        torch.cuda.empty_cache()
    out = run_dir / "induction_owner.json"
    out.write_text(json.dumps(
        {"base_copy2_ce": base_ce, "top_head": [int(x) for x in top_head],
         "ranking": [{"component": int(c), "score": float(weighted[c]),
                      "share_top_head": float(share_top[c])}
                     for c in order[:64]],
         "tests": results}, indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
