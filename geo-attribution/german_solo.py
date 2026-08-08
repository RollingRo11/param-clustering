"""Single-component per-matrix German edit: find the one component that works.

Phase 1 screens EVERY candidate component solo (per-matrix signed gains,
multilingual objective, one standard recipe). Phase 2 takes the top
candidates and sweeps the recipe (init basin: warm scalar / uniform
inversion / identity; lambda balance; lr; longer training). The winner gets
held-out evaluation, rollouts, and the literal-surgery check.
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
from german_vpd_1b import ce_each, log, prepare_data
from german_vpd_multi import ROMANCE, train_blocks
from german_permatrix import (COMPONENT_ORDER, INIT_SCALAR, PROMPTS,
                              PerMatrixEditor, language_metrics)


def train_solo(editor, data, slot, lam_en, lam_rom, lr, steps, init_kind,
               ceiling, tag, log_every=100):
    total = len(editor.components)
    modules = len(editor.modules)
    init = torch.ones(total, modules, device=editor.device)
    if init_kind == "warm":
        init[slot] = INIT_SCALAR[editor.components[slot]]
    elif init_kind == "invert":
        init[slot] = -12.0
    alpha = torch.nn.Parameter(init.clone())
    optimizer = torch.optim.Adam([alpha], lr=lr)
    blocks = train_blocks(data)
    base_logp = {}
    for lang in ("en",) + ROMANCE:
        rows = []
        for block in blocks[lang]:
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                base = editor.logits(block[None].to(editor.device), None)
            rows.append(F.log_softmax(base[:, :-1].float(), -1))
            del base
        base_logp[lang] = rows
    counts = {lang: len(blocks[lang]) for lang in blocks}
    keep = torch.zeros(total, 1, device=editor.device)
    keep[slot] = 1.0
    for step in range(steps):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in blocks}
        chosen["de"] = step % counts["de"]
        idx = torch.stack([blocks[lang][chosen[lang]] for lang in
                           ("de", "en", "fr", "es", "it")]).to(editor.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = editor.logits(idx, alpha)
            german_ce = ce_each(logits[:1], idx[:1]).squeeze(0)
            kls = []
            for j, lang in enumerate(("en",) + ROMANCE, start=1):
                edited_logp = F.log_softmax(
                    logits[j:j + 1, :-1].float(), -1)
                kls.append(F.kl_div(
                    edited_logp, base_logp[lang][chosen[lang]],
                    log_target=True, reduction="none").sum(-1).mean())
            romance = (kls[1] + kls[2] + kls[3]) / 3.0
            loss = (F.relu(ceiling - german_ce) + lam_en * kls[0]
                    + lam_rom * romance)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.nan_to_num_(nan=1.0, posinf=100.0, neginf=-50.0)
            alpha.clamp_(-50.0, 100.0)
            alpha.data = 1.0 + (alpha.data - 1.0) * keep  # only this slot
        if step % log_every == 0 or step + 1 == steps:
            log(f"{tag} step {step:04d}: deCE={german_ce.item():.2f} "
                f"enKL={kls[0].item():.3f} romKL={romance.item():.3f}")
        del logits
    return alpha.detach()


def evaluate(editor, data, alpha, cache):
    sets = {
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "codeparrot": data["code_eval"],
        "french_europarl_heldout": data["fr_eval"][2:],
        "spanish_europarl_heldout": data["es_eval"][2:],
        "italian_europarl_heldout": data["it_eval"][2:],
    }
    rows = {name: language_metrics(editor, idx, alpha, cache, name)
            for name, idx in sets.items()}
    romance = max(rows[n]["delta_ce"] for n in
                  ("french_europarl_heldout", "spanish_europarl_heldout",
                   "italian_europarl_heldout"))
    return ({n: round(m["delta_ce"], 4) for n, m in rows.items()},
            rows["german_europarl"]["delta_ce"],
            rows["english_pile"]["delta_ce"], romance)


def guarded_score(de, en, rom, en_budget=0.15, rom_budget=0.35):
    penalty = 10 * max(0.0, en - en_budget) + 4 * max(0.0, rom - rom_budget)
    return de - penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--screen_steps", type=int, default=400)
    parser.add_argument("--final_steps", type=int, default=800)
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=32)
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
    devices = args.devices or [
        f"cuda:{i}" for i in range(torch.cuda.device_count())]
    bank = torch.load(args.bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    target = geo1b.load_target_1b(devices[0])
    editor = PerMatrixEditor(target, bank, COMPONENT_ORDER, devices[0])
    del bank
    editors = [editor] + [editor.replicate(dev) for dev in devices[1:]]
    for twin in editors[1:]:
        for index, path in enumerate(twin.modules):
            twin.target.get_submodule(path)._module_index = index
    ceiling = math.log(editor.target.hf.config.vocab_size)
    total = len(COMPONENT_ORDER)

    # ---- Phase 1: screen every component solo, standard recipe ----
    def phase1_worker(ed, slots):
        cache = {}
        out = []
        for slot in slots:
            component = COMPONENT_ORDER[slot]
            tag = f"solo c{component}"
            alpha = train_solo(ed, data, slot, 10.0, 10.0, 0.3,
                               args.screen_steps, "warm", ceiling, tag,
                               log_every=args.screen_steps)
            detail, de, en, rom = evaluate(ed, data, alpha, cache)
            log(f"SOLO c{component}: de={de:+.2f} en={en:+.3f} "
                f"rom={rom:+.3f}")
            out.append({"slot": slot, "component": component, "phase": 1,
                        "recipe": "warm,l=10/10,lr=0.3", "de": de, "en": en,
                        "rom": rom, "detail": detail})
        return out

    slot_queues = [list(range(total))[i::len(editors)]
                   for i in range(len(editors))]
    results = []
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [pool.submit(phase1_worker, ed, q)
                   for ed, q in zip(editors, slot_queues) if q]
        for future in futures:
            results.extend(future.result())

    ranked = sorted(results,
                    key=lambda r: -guarded_score(r["de"], r["en"], r["rom"]))
    finalists = ranked[:args.top_n]
    log("finalists: " + ", ".join(
        f"c{r['component']} (de={r['de']:+.2f})" for r in finalists))

    # ---- Phase 2: recipe sweep on the finalists ----
    recipes = [
        ("warm", 10.0, 3.0, 0.3),
        ("warm", 10.0, 10.0, 0.1),
        ("warm", 30.0, 10.0, 0.3),
        ("invert", 10.0, 10.0, 0.3),
        ("identity", 10.0, 10.0, 0.3),
    ]
    jobs = [(r["slot"], r["component"], *recipe)
            for r in finalists for recipe in recipes]

    def phase2_worker(ed, queue):
        cache = {}
        out = []
        for slot, component, init_kind, lam_en, lam_rom, lr in queue:
            tag = (f"final c{component} {init_kind} l={lam_en:g}/{lam_rom:g} "
                   f"lr={lr:g}")
            alpha = train_solo(ed, data, slot, lam_en, lam_rom, lr,
                               args.final_steps, init_kind, ceiling, tag,
                               log_every=args.final_steps // 2)
            detail, de, en, rom = evaluate(ed, data, alpha, cache)
            log(f"FINAL c{component} [{init_kind},l={lam_en:g}/{lam_rom:g},"
                f"lr={lr:g}]: de={de:+.2f} en={en:+.3f} rom={rom:+.3f}")
            out.append({"slot": slot, "component": component, "phase": 2,
                        "recipe": f"{init_kind},l={lam_en:g}/{lam_rom:g},"
                                  f"lr={lr:g}",
                        "de": de, "en": en, "rom": rom, "detail": detail,
                        "alpha": alpha[slot].cpu()})
        return out

    job_queues = [jobs[i::len(editors)] for i in range(len(editors))]
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [pool.submit(phase2_worker, ed, q)
                   for ed, q in zip(editors, job_queues) if q]
        for future in futures:
            results.extend(future.result())

    finals = [r for r in results if r["phase"] == 2]
    best = max(finals, key=lambda r: guarded_score(r["de"], r["en"], r["rom"]))
    log(f"BEST single-component edit: c{best['component']} "
        f"[{best['recipe']}] de={best['de']:+.2f} en={best['en']:+.3f} "
        f"rom={best['rom']:+.3f}")

    # Rollouts + literal surgery for the winner on editor 0.
    editor0 = editors[0]
    full_alpha = torch.ones(total, len(editor0.modules),
                            device=editor0.device)
    full_alpha[best["slot"]] = best["alpha"].to(editor0.device)
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        expression = editor0.logits(
            data["de_eval"][:1, :32].to(editor0.device), full_alpha).float()
    saved = editor0.apply_in_place(full_alpha)
    editor0.alpha = None
    with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=True):
        literal = editor0.target(
            data["de_eval"][:1, :32].to(editor0.device)).float()
    error = (expression - literal).abs()
    verification = {"max_abs_logit_error": error.max().item(),
                    "mean_abs_logit_error": error.mean().item()}
    log(f"literal surgery: max |dlogit|="
        f"{verification['max_abs_logit_error']:.3e}")

    def generate(prompt):
        ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=editor0.device)
        for _ in range(args.max_new_tokens):
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=True):
                logits = editor0.target(ids[:, -512:])
            ids = torch.cat(
                [ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return tokenizer.decode(ids[0].tolist())

    rollouts = {lang: {"prompt": p, "edited": generate(p)}
                for lang, p in PROMPTS.items()}
    editor0.restore(saved)

    output = args.run_dir / "german_solo.json"
    output.write_text(json.dumps({
        "format": "german_solo_v1",
        "component_pool": COMPONENT_ORDER,
        "phase1": [{k: v for k, v in r.items() if k != "alpha"}
                   for r in results if r["phase"] == 1],
        "phase2": [{k: v for k, v in r.items() if k != "alpha"}
                   for r in finals],
        "best": {k: v for k, v in best.items() if k != "alpha"},
        "literal_weight_edit_verification": verification,
        "rollouts": rollouts,
    }, indent=2))
    torch.save({
        "format": "solo_permatrix_adapter_v1",
        "model": geo1b.model_identity(),
        "bank": str(args.bank_path),
        "component": best["component"],
        "recipe": best["recipe"],
        "alpha_per_matrix": best["alpha"],
        "modules_order": editor0.modules,
    }, args.run_dir / "german_solo_adapter.pt")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
