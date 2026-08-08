"""Token-budget race: k=8 per-matrix component edit vs LoRA r=1.

Recreates the LessWrong post's experiment at 1B: for each German training
token budget B, fine-tune (a) the decomposition-native k=8 per-matrix
component edit and (b) plain LoRA r=1, both under the multilingual objective,
and measure held-out German removal vs English/Romance leak. The German
training data is truncated to exactly B tokens; preserve-language data stays
full, matching the post (the budget scarcity is German evidence).

Arms:
  comp_warm  - k=8 per-matrix, init from the method's component priors
  comp_cold  - k=8 per-matrix, identity init (control)
  lora       - plain rank-1 LoRA on all 112 matrices
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import geo1b  # noqa: F401
import geo67
from german_vpd_1b import ce_each, log, prepare_data
from german_vpd_multi import ROMANCE, train_blocks
from german_lora_guided import GuidedLora
from german_permatrix import (COMPONENT_ORDER, INIT_SCALAR, PerMatrixEditor,
                              language_metrics)

K = 8


def german_row(data, budget: int, device: str) -> torch.Tensor:
    flat = data["de_train"].flatten()
    if budget > flat.numel():
        raise ValueError(f"budget {budget} exceeds German train tokens")
    return flat[:budget][None].to(device)


def preserve_logp(model_logits, blocks, device):
    out = {}
    for lang in ("en",) + ROMANCE:
        rows = []
        for block in blocks[lang]:
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                base = model_logits(block[None].to(device))
            rows.append(F.log_softmax(base[:, :-1].float(), -1))
            del base
        out[lang] = rows
    return out


def objective_terms(logits_de, de_row, logits_pres, base_rows, lam_en,
                    lam_rom, ceiling):
    german_ce = ce_each(logits_de, de_row).squeeze(0)
    kls = []
    for j in range(4):
        edited_logp = F.log_softmax(logits_pres[j:j + 1, :-1].float(), -1)
        kls.append(F.kl_div(edited_logp, base_rows[j], log_target=True,
                            reduction="none").sum(-1).mean())
    romance = (kls[1] + kls[2] + kls[3]) / 3.0
    loss = F.relu(ceiling - german_ce) + lam_en * kls[0] + lam_rom * romance
    return loss, german_ce


def run_comp(editor, data, budget, lam_en, lam_rom, lr, steps, ceiling,
             warm, cache, tag):
    total = len(editor.components)
    modules = len(editor.modules)
    init = torch.ones(total, modules, device=editor.device)
    if warm:
        for slot in range(K):
            init[slot] = INIT_SCALAR[editor.components[slot]]
    alpha = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.Adam([alpha], lr=lr)
    blocks = train_blocks(data)
    base_logp = preserve_logp(
        lambda idx: editor.logits(idx, None), blocks, editor.device)
    de_row = german_row(data, budget, editor.device)
    counts = {lang: len(blocks[lang]) for lang in blocks}
    t0 = time.perf_counter()
    for step in range(steps):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in ("en",) + ROMANCE}
        pres = torch.stack(
            [blocks[lang][chosen[lang]]
             for lang in ("en",) + ROMANCE]).to(editor.device)
        base_rows = [base_logp[lang][chosen[lang]]
                     for lang in ("en",) + ROMANCE]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits_de = editor.logits(de_row, alpha)
            logits_pres = editor.logits(pres, alpha)
            loss, german_ce = objective_terms(
                logits_de, de_row, logits_pres, base_rows, lam_en, lam_rom,
                ceiling)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.nan_to_num_(nan=1.0, posinf=100.0, neginf=-50.0)
            alpha.clamp_(-50.0, 100.0)
            alpha[K:] = 1.0
    result = evaluate_alpha(editor, data, alpha.detach(), cache)
    result["train_seconds"] = time.perf_counter() - t0
    log(f"{tag}: de={result['german']:+.2f} en={result['english']:+.3f} "
        f"rom={result['romance']:+.3f}")
    return result


def evaluate_alpha(editor, data, alpha, cache):
    sets = {"german": data["de_eval"], "english": data["pile_en_eval"],
            "fr": data["fr_eval"][2:], "es": data["es_eval"][2:],
            "it": data["it_eval"][2:]}
    rows = {name: language_metrics(editor, idx, alpha, cache, name)
            for name, idx in sets.items()}
    return {"german": rows["german"]["delta_ce"],
            "english": rows["english"]["delta_ce"],
            "romance": max(rows[l]["delta_ce"] for l in ("fr", "es", "it")),
            "detail": {n: round(m["delta_ce"], 4) for n, m in rows.items()}}


def lora_metrics(model, idx, cache, key):
    idx = idx.to(model.device)
    if key not in cache:
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=True):
            base = model.logits(idx, False)
        cache[key] = (ce_each(base, idx).mean().item(),
                      F.log_softmax(base[:, :-1].float(), -1))
        del base
    base_ce, base_logp = cache[key]
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        edited = model.logits(idx, True)
    edited_ce = ce_each(edited, idx).mean().item()
    return edited_ce - base_ce


def run_lora(model, data, budget, lam_en, lam_rom, lr, steps, ceiling,
             cache, tag):
    model.reset()
    optimizer = torch.optim.Adam(model.params, lr=lr)
    blocks = train_blocks(data)
    base_logp = preserve_logp(
        lambda idx: model.logits(idx, False), blocks, model.device)
    de_row = german_row(data, budget, model.device)
    counts = {lang: len(blocks[lang]) for lang in blocks}
    t0 = time.perf_counter()
    for step in range(steps):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in ("en",) + ROMANCE}
        pres = torch.stack(
            [blocks[lang][chosen[lang]]
             for lang in ("en",) + ROMANCE]).to(model.device)
        base_rows = [base_logp[lang][chosen[lang]]
                     for lang in ("en",) + ROMANCE]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits_de = model.logits(de_row, True)
            logits_pres = model.logits(pres, True)
            loss, german_ce = objective_terms(
                logits_de, de_row, logits_pres, base_rows, lam_en, lam_rom,
                ceiling)
        loss.backward()
        optimizer.step()
    sets = {"german": data["de_eval"], "english": data["pile_en_eval"],
            "fr": data["fr_eval"][2:], "es": data["es_eval"][2:],
            "it": data["it_eval"][2:]}
    rows = {name: lora_metrics(model, idx, cache, name)
            for name, idx in sets.items()}
    result = {"german": rows["german"], "english": rows["english"],
              "romance": max(rows[l] for l in ("fr", "es", "it")),
              "detail": {n: round(v, 4) for n, v in rows.items()},
              "train_seconds": time.perf_counter() - t0}
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
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
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

    results = []

    def comp_worker():
        bank = torch.load(args.bank_path, weights_only=True,
                          map_location="cpu", mmap=True)
        target = geo1b.load_target_1b("cuda:0")
        editor = PerMatrixEditor(target, bank, COMPONENT_ORDER, "cuda:0")
        del bank
        cache = {}
        rows = []
        for budget in args.budgets:
            for lam_en, lam_rom in ((10.0, 10.0), (30.0, 30.0)):
                for lr in (0.1, 0.3):
                    tag = (f"comp_warm B={budget} l={lam_en:g}/{lam_rom:g} "
                           f"lr={lr:g}")
                    row = run_comp(editor, data, budget, lam_en, lam_rom, lr,
                                   args.steps, ceiling, True, cache, tag)
                    rows.append({"arm": "comp_warm", "budget": budget,
                                 "lam_en": lam_en, "lam_rom": lam_rom,
                                 "lr": lr, **row})
            tag = f"comp_cold B={budget} l=10/10 lr=0.1"
            row = run_comp(editor, data, budget, 10.0, 10.0, 0.1,
                           args.steps, ceiling, False, cache, tag)
            rows.append({"arm": "comp_cold", "budget": budget,
                         "lam_en": 10.0, "lam_rom": 10.0, "lr": 0.1, **row})
        return rows

    def lora_worker():
        target = geo1b.load_target_1b("cuda:1")
        model = GuidedLora(target, geo67.MODULES, 1, "cuda:1", args.seed,
                           masks=None)
        cache = {}
        rows = []
        for budget in args.budgets:
            for lam_en, lam_rom in ((10.0, 10.0), (30.0, 30.0)):
                for lr in (3e-3, 1e-2):
                    tag = (f"lora B={budget} l={lam_en:g}/{lam_rom:g} "
                           f"lr={lr:g}")
                    row = run_lora(model, data, budget, lam_en, lam_rom, lr,
                                   args.steps, ceiling, cache, tag)
                    rows.append({"arm": "lora", "budget": budget,
                                 "lam_en": lam_en, "lam_rom": lam_rom,
                                 "lr": lr, **row})
        return rows

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(comp_worker), pool.submit(lora_worker)]
        for future in futures:
            results.extend(future.result())

    output = args.run_dir / "budget_race.json"
    output.write_text(json.dumps({
        "format": "budget_race_v1",
        "budgets": args.budgets,
        "chance_delta": ceiling - 2.664,
        "k": K,
        "components": COMPONENT_ORDER[:K],
        "steps": args.steps,
        "results": results,
    }, indent=2))
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
