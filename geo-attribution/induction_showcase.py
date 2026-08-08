"""A clean text demonstration of the induction edit.

The earlier examples were over-determined: "4-hydroxybenz" -> "aldehyde" is
predictable from ordinary language statistics, so knocking out induction does
not stop it. To see the mechanism fail you need continuations the model cannot
possibly know from pretraining — novel serial numbers, invented names, random
token strings — where copying an earlier occurrence is the ONLY route.

Paired against control prompts that require no copying at all, so the claim
"gibberish on copying, normal otherwise" is visible side by side rather than
asserted.

  python3.12 induction_showcase.py --alphas 8 12 16 --component 3392
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

# (prompt, the continuation only obtainable by copying)
COPY = [
    ("Serial: 8QK4T2M91XZ. Please confirm the serial: 8QK", "4T2M91XZ"),
    ("The wizard Gorblexi Thundermaw cast a spell. Later, Gorblexi Thunder",
     "maw"),
    ("x7q4 m2p9 k8w1 x7q4 m2p9", " k8w1"),
    ("Key = 'Vh3nQpZr'. Reusing Key = 'Vh3n", "QpZr"),
    ("def zqxvbnmplorktz(a): return a\nresult = zqxvbnm", "plorktz"),
    ("Token: RJ7-WWQ-2LM. Repeat token: RJ7-", "WWQ-2LM"),
]

# no copying required; ordinary language modelling
CONTROL = [
    "The capital of France is",
    "Water boils at one hundred degrees",
    "She opened the door and walked into the",
    "The main advantage of solar power is that it",
    "def add(a, b):\n    return",
    "In 1969, humans first landed on the",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--component", type=int, default=3392)
    ap.add_argument("--alphas", type=float, nargs="+", default=[8, 12, 16])
    ap.add_argument("--new_tokens", type=int, default=14)
    ap.add_argument("--span", type=int, default=64)
    ap.add_argument("--out", default="induction_showcase.json")
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
    S = args.span
    idx = repeated_batch(8, S, 1000, 20000, cfg.bos_token_id, dev, 0)
    second = torch.zeros_like(idx, dtype=torch.bool)
    second[:, S + 9:2 * S] = True
    ctrl_blocks = control_batch(tok, 96, dev)
    cm = torch.ones_like(ctrl_blocks, dtype=torch.bool)
    cm[:, :4] = False

    def greedy(text):
        ids = torch.tensor([tok.encode(text)], device=dev)
        for _ in range(args.new_tokens):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=True):
                lg = hf(ids[:, -512:]).logits
            ids = torch.cat([ids, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return tok.decode(ids[0, -args.new_tokens:].tolist())

    def ces():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=True):
            a = ce_at(hf(idx).logits, idx, second)
            b = ce_at(hf(ctrl_blocks).logits, ctrl_blocks, cm)
        return a, b

    def measure(label):
        copies = [(p, want, greedy(p)) for p, want in COPY]
        hits = sum(1 for _, want, got in copies
                   if got.strip().startswith(want.strip()[:len(want.strip())]))
        ctrls = [(p, greedy(p)) for p in CONTROL]
        d2, cc = ces()
        log(f"--- {label}: copy-task {hits}/{len(COPY)} correct | "
            f"probe CE {d2:.3f} | ordinary-text CE {cc:.3f}")
        for p, want, got in copies:
            mark = "OK " if got.strip().startswith(want.strip()) else "MISS"
            log(f"  [{mark}] ...{p[-30:]!r} want {want!r} -> {got!r}")
        for p, got in ctrls:
            log(f"  [ctl] ...{p[-30:]!r} -> {got!r}")
        return {"label": label, "copy_hits": hits, "copy_total": len(COPY),
                "probe_ce": d2, "control_ce": cc,
                "copy": [{"prompt": p, "want": w, "got": g}
                         for p, w, g in copies],
                "control": [{"prompt": p, "got": g} for p, g in ctrls]}

    out = {"component": args.component, "arms": [measure("base")]}
    ed = PerMatrixEditor(target, bank, [args.component], dev)
    nm = len(ed.modules)
    for a in args.alphas:
        saved = ed.apply_in_place(torch.full((1, nm), float(a), device=dev))
        ed.alpha = None
        out["arms"].append(measure(f"c{args.component} alpha={a:g}"))
        ed.restore(saved)
    path = run_dir / args.out
    path.write_text(json.dumps(out, indent=1))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
