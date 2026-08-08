"""Reproduce the best single-component German edit and read what it writes.

The operating point is the one the fair sweep selects for the component arm on
the english-only objective: c3634, 2048 German tokens, lr 0.3, lambda_en 100,
100 Adam steps over 112 per-matrix scalars. Held-out scores at that point are
German +8.90 nats against a chance ceiling of +9.10, English +0.016.

CE numbers are the evidence; completions are how anyone reads them. So this
trains the edit, verifies it landed on the same operating point, and then
greedily decodes matched German and English prompts through the same weights.
Same model, same decode, one component scaled.

    python3.12 german_completions.py --new_tokens 40
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
import budget_race as br
from german_vpd_1b import log, prepare_data
from german_permatrix import PerMatrixEditor
from lora_fair_sweep import make_sets, summarize, SOLO, INVERT_INIT

GERMAN = [
    "Berlin ist die Hauptstadt von",
    "Die Bundesregierung hat am Montag beschlossen, dass",
    "Der Zweite Weltkrieg endete im Jahr",
    "Ich möchte einen Kaffee mit Milch und",
    "Das Wetter in München ist heute",
    "Die wichtigsten Werke von Johann Wolfgang von Goethe sind",
]
ENGLISH = [
    "Berlin is the capital of",
    "The government announced on Monday that",
    "The Second World War ended in the year",
    "I would like a coffee with milk and",
    "The weather in Munich today is",
    "The most important works of Johann Wolfgang von Goethe are",
]
FRENCH = [
    "Berlin est la capitale de",
    "Le gouvernement a annoncé lundi que",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--component", type=int, default=SOLO)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--lam_en", type=float, default=100.0)
    ap.add_argument("--lam_rom", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--new_tokens", type=int, default=40)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--train_tokens", type=int, default=2048)
    ap.add_argument("--eval_blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh_data", action="store_true")
    ap.add_argument("--out", default="german_completions.json")
    args = ap.parse_args()
    dev = args.device
    if dev.startswith("cuda:"):
        torch.cuda.set_device(int(dev.split(":")[1]))
    args.run_dir = args.artifact_root / args.tag
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tok)
    report, devset = make_sets(data)
    ceiling = math.log(128256)

    bank = torch.load(args.run_dir / f"banks_{args.banks_tag}.pt",
                      weights_only=True, map_location="cpu", mmap=True)
    target = geo1b.load_target_1b(dev)
    ed = PerMatrixEditor(target, bank, [args.component], dev)
    del bank
    n_mod = len(ed.modules)
    log(f"c{args.component}: {n_mod} scalars, mass fraction "
        f"{ed.mass_fraction[0]:.5f}")

    # ---- train the edit, exactly the swept recipe ----
    alpha = torch.nn.Parameter(torch.full((1, n_mod), INVERT_INIT, device=dev))
    opt = torch.optim.Adam([alpha], lr=args.lr)
    blocks = br.train_blocks(data)
    base_logp = br.preserve_logp(lambda idx: ed.logits(idx, None), blocks, dev)
    de_row = br.german_row(data, args.budget, dev)
    counts = {lang: len(blocks[lang]) for lang in blocks}
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in ("en",) + br.ROMANCE}
        pres = torch.stack([blocks[lang][chosen[lang]]
                            for lang in ("en",) + br.ROMANCE]).to(dev)
        base_rows = [base_logp[lang][chosen[lang]]
                     for lang in ("en",) + br.ROMANCE]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss, _ = br.objective_terms(
                ed.logits(de_row, alpha), de_row, ed.logits(pres, alpha),
                base_rows, args.lam_en, args.lam_rom, ceiling)
        loss.backward()
        opt.step()
        with torch.no_grad():
            alpha.nan_to_num_(nan=1.0, posinf=100.0, neginf=-50.0)
            alpha.clamp_(-50.0, 100.0)
    a = alpha.detach()
    log(f"trained {args.steps} steps in {time.perf_counter() - t0:.0f}s")

    cache = {}
    ev = summarize({n: br.language_metrics(ed, idx, a, cache, n)["delta_ce"]
                    for n, idx in report.items()})
    log(f"held-out ΔCE: german {ev['german']:+.2f} (chance "
        f"{ceiling - 2.664:+.2f}) | english {ev['detail']['english']:+.3f} "
        f"| fr {ev['detail']['fr']:+.3f} es {ev['detail']['es']:+.3f} "
        f"it {ev['detail']['it']:+.3f}")

    # ---- greedy completions, base vs edited, same decode ----
    def greedy(text, n):
        ids = torch.tensor([tok.encode(text)], device=dev)
        for _ in range(n):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=True):
                lg = target(ids[:, -512:])
            ids = torch.cat([ids, lg[:, -1].argmax(-1, keepdim=True)], 1)
        return tok.decode(ids[0, -n:].tolist())

    sets = [("German", GERMAN), ("English", ENGLISH), ("French", FRENCH)]
    out = {"component": args.component, "n_scalars": n_mod,
           "config": {"budget": args.budget, "lr": args.lr,
                      "lam_en": args.lam_en, "lam_rom": args.lam_rom,
                      "steps": args.steps},
           "eval": ev, "chance_delta": ceiling - 2.664, "completions": []}

    ed.alpha = None
    base_txt = {name: [greedy(p, args.new_tokens) for p in ps]
                for name, ps in sets}
    saved = ed.apply_in_place(a)          # literal weights, no hooks
    ed.alpha = None
    edit_txt = {name: [greedy(p, args.new_tokens) for p in ps]
                for name, ps in sets}
    ed.restore(saved)

    for name, ps in sets:
        log(f"\n================  {name}  ================")
        for i, p in enumerate(ps):
            out["completions"].append({
                "language": name, "prompt": p,
                "base": base_txt[name][i], "edited": edit_txt[name][i]})
            log(f"\n  prompt   {p!r}")
            log(f"  base   → {base_txt[name][i]!r}")
            log(f"  edited → {edit_txt[name][i]!r}")

    path = args.run_dir / args.out
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(f"\nwrote {path}")


if __name__ == "__main__":
    main()
