"""Single-component edit vs LoRA r=1 across German token budgets, under two
preserve objectives.

  multilingual : relu(logV - de_CE) + 10*KL_en + 10*mean KL_{fr,es,it}
  english-only : relu(logV - de_CE) + 10*KL_en          <- Romance unprotected

The english-only arm is the interesting one: with nothing defending French,
Spanish and Italian, does the free-form LoRA or the decomposition-native edit
do more collateral damage while removing the same German?

Both arms get identical data, budgets, step count and objective; the component
arm is c3634 alone (112 per-matrix gains, invert init, the recipe that won the
solo search).

  python3.12 budget_race_solo.py --steps 800
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import geo1b  # noqa: F401
import geo67
import budget_race as br
from german_vpd_1b import log, prepare_data
from german_lora_guided import GuidedLora
from german_permatrix import PerMatrixEditor

SOLO = 3634
INVERT_INIT = -12.0


def run_solo(editor, data, budget, lam_en, lam_rom, lr, steps, ceiling,
             cache, tag):
    """run_comp's loop with the solo recipe: one component, inverted init."""
    n_mod = len(editor.modules)
    init = torch.full((1, n_mod), INVERT_INIT, device=editor.device)
    alpha = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.Adam([alpha], lr=lr)
    blocks = br.train_blocks(data)
    base_logp = br.preserve_logp(
        lambda idx: editor.logits(idx, None), blocks, editor.device)
    de_row = br.german_row(data, budget, editor.device)
    counts = {lang: len(blocks[lang]) for lang in blocks}
    t0 = time.perf_counter()
    for step in range(steps):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in ("en",) + br.ROMANCE}
        pres = torch.stack(
            [blocks[lang][chosen[lang]]
             for lang in ("en",) + br.ROMANCE]).to(editor.device)
        base_rows = [base_logp[lang][chosen[lang]]
                     for lang in ("en",) + br.ROMANCE]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits_de = editor.logits(de_row, alpha)
            logits_pres = editor.logits(pres, alpha)
            loss, _ = br.objective_terms(logits_de, de_row, logits_pres,
                                         base_rows, lam_en, lam_rom, ceiling)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.nan_to_num_(nan=1.0, posinf=100.0, neginf=-50.0)
            alpha.clamp_(-50.0, 100.0)
    result = br.evaluate_alpha(editor, data, alpha.detach(), cache)
    result["train_seconds"] = time.perf_counter() - t0
    log(f"{tag}: de={result['german']:+.2f} en={result['english']:+.3f} "
        f"rom={result['romance']:+.3f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[8, 64, 512, 2048])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
    parser.add_argument("--out", default="budget_race_solo.json")
    args = parser.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tokenizer)
    ceiling = math.log(128256)
    # (name, lam_en, lam_rom)
    OBJECTIVES = [("multilingual", 10.0, 10.0), ("english_only", 10.0, 0.0)]

    # Models are loaded inside each worker BEFORE any other thread runs a
    # forward; HF from_pretrained is not thread-safe, so both loads are done
    # up front on the main thread.
    bank = torch.load(args.bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    target0 = geo1b.load_target_1b("cuda:0")
    editor = PerMatrixEditor(target0, bank, [SOLO], "cuda:0")
    del bank
    target1 = geo1b.load_target_1b("cuda:1")
    lora = GuidedLora(target1, geo67.MODULES, 1, "cuda:1", args.seed,
                      masks=None)

    def comp_worker():
        cache, rows = {}, []
        for obj, lam_en, lam_rom in OBJECTIVES:
            for budget in args.budgets:
                tag = f"solo c{SOLO} {obj} B={budget}"
                row = run_solo(editor, data, budget, lam_en, lam_rom, 0.3,
                               args.steps, ceiling, cache, tag)
                rows.append({"arm": "component", "component": SOLO,
                             "objective": obj, "budget": budget,
                             "lam_en": lam_en, "lam_rom": lam_rom,
                             "lr": 0.3, **row})
        return rows

    def lora_worker():
        cache, rows = {}, []
        for obj, lam_en, lam_rom in OBJECTIVES:
            for budget in args.budgets:
                for lr in (3e-3, 1e-2):
                    tag = f"lora {obj} B={budget} lr={lr:g}"
                    row = br.run_lora(lora, data, budget, lam_en, lam_rom, lr,
                                      args.steps, ceiling, cache, tag)
                    rows.append({"arm": "lora", "objective": obj,
                                 "budget": budget, "lam_en": lam_en,
                                 "lam_rom": lam_rom, "lr": lr, **row})
        return rows

    results = []
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(comp_worker), pool.submit(lora_worker)]
        for f in futures:
            results.extend(f.result())
    out = args.run_dir / args.out
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out} ({len(results)} arms)")


if __name__ == "__main__":
    main()
