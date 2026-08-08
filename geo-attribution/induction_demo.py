"""Text examples + sweep data for the induction component edits.

Two components do very different things to induction, and the point of this
script is to show it in text as well as in numbers:

  c3392  breaks the MECHANISM  — the induction heads stop attending to the
         copy-1 successor, so the model can no longer copy anything.
  c108   breaks the READOUT    — the attention pattern is untouched; the model
         still looks in the right place and still fails to say the word.

Prompts are natural-text copying tasks (a rare name, a serial number, a long
identifier seen once already), which is what induction does outside a synthetic
repeated-token probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import geo1b  # noqa: F401
from german_permatrix import PerMatrixEditor
from german_vpd_1b import log
from induction4096 import repeated_batch, control_batch, ce_at

PROMPTS = [
    "The access code is QZ4471. Please write it down. The access code is QZ",
    "Mrs. Wexlerbaum arrived at noon. A little later, Mrs. Wexler",
    "import krylov_subspace_solver\nresult = krylov_subspace",
    "apple banana cherry apple banana",
    "Dr. Anantharaman published in 1998. In 2001, Dr. Ananth",
    "The compound is 4-hydroxybenzaldehyde. We dissolved the 4-hydroxybenz",
]

ARMS = [("base", None, None),
        ("c3392 (circuit) a=8", 3392, 8.0),
        ("c3392 (circuit) a=16", 3392, 16.0),
        ("c108 (readout) a=12", 108, 12.0),
        ("c108 (readout) a=16", 108, 16.0),
        ("c2747 (inert) a=16", 2747, 16.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--span", type=int, default=64)
    ap.add_argument("--n_seq", type=int, default=8)
    ap.add_argument("--new_tokens", type=int, default=12)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0, 2, 4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--skip_sweeps", action="store_true")
    ap.add_argument("--out", default="induction_demo.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    run_dir = args.artifact_root / args.tag

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    bank = torch.load(run_dir / f"banks_{args.banks_tag}.pt", weights_only=True,
                      map_location="cpu", mmap=True)
    target = geo1b.load_target_1b(dev)
    hf, cfg = target.hf, target.hf.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    S = args.span
    idx = repeated_batch(args.n_seq, S, 1000, 20000, cfg.bos_token_id, dev, 0)
    second = torch.zeros_like(idx, dtype=torch.bool)
    second[:, S + 9:2 * S] = True
    pos = torch.arange(S + 9, 2 * S, device=dev)
    tgt = pos - S + 1
    ctrl = control_batch(tok, 96, dev)
    cm = torch.ones_like(ctrl, dtype=torch.bool)
    cm[:, :4] = False

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

    def ces():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=True):
            a = ce_at(hf(idx).logits, idx, second)
            b = ce_at(hf(ctrl).logits, ctrl, cm)
        return a, b

    def greedy(text):
        ids = torch.tensor([tok.encode(text)], device=dev)
        for _ in range(args.new_tokens):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=True):
                lg = hf(ids[:, -512:]).logits
            ids = torch.cat([ids, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return tok.decode(ids[0, -args.new_tokens:].tolist())

    base_sc = head_scores()
    b2, bc = ces()
    top12 = np.argsort(-base_sc.reshape(-1))[:12]
    log(f"base: copy2 CE {b2:.3f}  control CE {bc:.3f}  "
        f"L10H23 {base_sc[10, 23]:.3f}")

    results = {"base": {"copy2_ce": b2, "control_ce": bc,
                        "l10h23": float(base_sc[10, 23]),
                        "top12_mean": float(base_sc.reshape(-1)[top12].mean())},
               "prompts": PROMPTS, "arms": [], "sweeps": {}}

    # ---- text examples ----
    for name, comp, alpha in ARMS:
        if comp is None:
            gens = [greedy(p) for p in PROMPTS]
            results["arms"].append({"name": name, "component": None,
                                    "alpha": None, "gens": gens,
                                    "copy2_ce": b2, "control_ce": bc,
                                    "l10h23": float(base_sc[10, 23])})
            for p, g in zip(PROMPTS, gens):
                log(f"[{name}] {p[-38:]!r} -> {g!r}")
            continue
        ed = PerMatrixEditor(target, bank, [comp], dev)
        nm = len(ed.modules)
        saved = ed.apply_in_place(torch.full((1, nm), alpha, device=dev))
        ed.alpha = None
        gens = [greedy(p) for p in PROMPTS]
        sc = head_scores()
        d2, cc = ces()
        ed.restore(saved)
        results["arms"].append({
            "name": name, "component": comp, "alpha": alpha, "gens": gens,
            "copy2_ce": d2, "control_ce": cc, "l10h23": float(sc[10, 23]),
            "top12_mean": float(sc.reshape(-1)[top12].mean())})
        log(f"[{name}] copy2 {d2:.3f}  ctrl {cc:.3f}  "
            f"L10H23 {base_sc[10,23]:.3f}->{sc[10,23]:.3f}")
        for p, g in zip(PROMPTS, gens):
            log(f"[{name}] {p[-38:]!r} -> {g!r}")
        del ed
        torch.cuda.empty_cache()

    # ---- alpha sweeps for the dissociation plot ----
    for comp in ([] if args.skip_sweeps else (3392, 108, 2747)):
        ed = PerMatrixEditor(target, bank, [comp], dev)
        nm = len(ed.modules)
        curve = []
        for a in args.alphas:
            saved = ed.apply_in_place(torch.full((1, nm), float(a), device=dev))
            ed.alpha = None
            sc = head_scores()
            d2, cc = ces()
            ed.restore(saved)
            curve.append({"alpha": a, "copy2_ce": d2, "control_ce": cc,
                          "l10h23": float(sc[10, 23]),
                          "top12_mean": float(sc.reshape(-1)[top12].mean())})
            log(f"  c{comp} a={a:g}: copy2 {d2:7.3f}  ctrl {cc:6.3f}  "
                f"L10H23 {sc[10,23]:.3f}")
        results["sweeps"][str(comp)] = curve
        del ed
        torch.cuda.empty_cache()

    path = run_dir / args.out
    path.write_text(json.dumps(results, indent=1))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
