"""Pareto frontier of German removal vs collateral: one component vs LoRA r=1.

Why not a single matched configuration: 112 scalars at lr 0.3 and 704k LoRA
parameters at lr 3e-4 have no reason to converge in the same number of steps.
Fixing both at 800 overtrains LoRA past the point where the objective's relu
term stops rewarding German damage, which reads as "LoRA is indiscriminate"
when it is really "LoRA was given the wrong schedule". So each method is swept
over (lr, lambda, step) and judged by its achievable frontier, and the two are
compared at MATCHED GERMAN REMOVAL rather than matched compute.

Checkpoints are scored twice:
  dev   — de_dev / en_dev and the Romance "selection guard" block that the
          protocol reserves for exactly this purpose. Used to CHOOSE points.
  eval  — strictly held-out blocks (Romance blocks 2+, English from the Pile).
          Used to REPORT them.
Selecting and reporting on the same blocks would flatter whichever method has
more hyperparameters to tune, which is LoRA.

  python3.12 lora_fair_sweep.py --objectives english_only --budgets 2048 512 64 8
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
EVAL_STEPS = [25, 50, 100, 200, 400, 800]
# LoRA converges in ~100 steps and then drifts off its own objective; the
# 112-scalar component edit needs the full 800. Giving each arm the steps it
# actually uses is both fairer than a shared cap and much cheaper than one.
ARM_MAX_STEPS = {"lora": 200, "component": 800}


def make_sets(data):
    """(reported, selection) block sets — disjoint by construction."""
    report = {"german": data["de_eval"], "english": data["pile_en_eval"],
              "fr": data["fr_eval"][2:], "es": data["es_eval"][2:],
              "it": data["it_eval"][2:]}
    dev = {"d_german": data["de_dev"], "d_english": data["en_dev"],
           "d_fr": data["fr_eval"][1:2], "d_es": data["es_eval"][1:2],
           "d_it": data["it_eval"][1:2]}
    return report, dev


def summarize(rows, prefix=""):
    g, e = rows[f"{prefix}german"], rows[f"{prefix}english"]
    rom = max(rows[f"{prefix}{l}"] for l in ("fr", "es", "it"))
    return {"german": g, "english": e, "romance": rom,
            "detail": {n: round(v, 4) for n, v in rows.items()}}


def run_sweep(kind, model, data, budget, lam_en, lam_rom, lr, ceiling,
              cache, tag):
    report, dev = make_sets(data)
    is_comp = kind == "component"
    if is_comp:
        n_mod = len(model.modules)
        alpha = torch.nn.Parameter(
            torch.full((1, n_mod), INVERT_INIT, device=model.device))
        params = [alpha]
        fwd_base = lambda idx: model.logits(idx, None)
    else:
        model.reset()
        alpha = None
        params = model.params
        fwd_base = lambda idx: model.logits(idx, False)
    opt = torch.optim.Adam(params, lr=lr)
    blocks = br.train_blocks(data)
    base_logp = br.preserve_logp(fwd_base, blocks, model.device)
    de_row = br.german_row(data, budget, model.device)
    counts = {lang: len(blocks[lang]) for lang in blocks}
    cap_steps = ARM_MAX_STEPS[kind]
    steps = [s for s in EVAL_STEPS if s <= cap_steps]
    out, t0 = [], time.perf_counter()
    for step in range(1, cap_steps + 1):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in ("en",) + br.ROMANCE}
        pres = torch.stack([blocks[lang][chosen[lang]]
                            for lang in ("en",) + br.ROMANCE]).to(model.device)
        base_rows = [base_logp[lang][chosen[lang]]
                     for lang in ("en",) + br.ROMANCE]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            if is_comp:
                lg_de = model.logits(de_row, alpha)
                lg_pr = model.logits(pres, alpha)
            else:
                lg_de = model.logits(de_row, True)
                lg_pr = model.logits(pres, True)
            loss, _ = br.objective_terms(lg_de, de_row, lg_pr, base_rows,
                                         lam_en, lam_rom, ceiling)
        loss.backward()
        opt.step()
        if is_comp:
            with torch.no_grad():
                alpha.nan_to_num_(nan=1.0, posinf=100.0, neginf=-50.0)
                alpha.clamp_(-50.0, 100.0)
        if step in steps:
            def score(sets):
                if is_comp:
                    return {n: br.language_metrics(model, idx, alpha.detach(),
                                                   cache, n)["delta_ce"]
                            for n, idx in sets.items()}
                return {n: br.lora_metrics(model, idx, cache, n)
                        for n, idx in sets.items()}
            ev = summarize(score(report))
            dv = summarize(score(dev), prefix="d_")
            out.append({"arm": kind, "step": step, "lr": lr, "lam_en": lam_en,
                        "lam_rom": lam_rom, "budget": budget,
                        "eval": ev, "dev": dv,
                        "seconds": time.perf_counter() - t0})
            log(f"{tag} step {step}: eval de={ev['german']:+.2f} "
                f"en={ev['english']:+.3f} rom={ev['romance']:+.3f} | "
                f"dev de={dv['german']:+.2f} rom={dv['romance']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--banks_tag", default="prop1b")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--train_tokens", type=int, default=2048)
    ap.add_argument("--eval_blocks", type=int, default=4)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2048, 512, 64, 8])
    ap.add_argument("--objectives", nargs="+",
                    default=["english_only", "multilingual"])
    ap.add_argument("--lora_lrs", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--comp_lrs", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[10.0, 100.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh_data", action="store_true")
    ap.add_argument("--out", default="lora_fair_sweep.json")
    args = ap.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(geo1b.MODEL_ID,
                                        revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tok)
    ceiling = math.log(128256)
    OBJ = {"multilingual": 1.0, "english_only": 0.0}

    bank = torch.load(args.bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    t0 = geo1b.load_target_1b("cuda:0")
    editor = PerMatrixEditor(t0, bank, [SOLO], "cuda:0")
    del bank
    t1 = geo1b.load_target_1b("cuda:1")
    lora = GuidedLora(t1, geo67.MODULES, 1, "cuda:1", args.seed, masks=None)

    def worker(kind, model, lrs):
        cache, rows = {}, []
        for obj in args.objectives:
            for B in args.budgets:
                for lam in args.lambdas:
                    for lr in lrs:
                        tag = f"{kind} {obj} B={B} l={lam:g} lr={lr:g}"
                        rows += [dict(objective=obj, **r) for r in
                                 run_sweep(kind, model, data, B, lam,
                                           lam * OBJ[obj], lr, ceiling,
                                           cache, tag)]
        return rows

    results = []
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(worker, "component", editor, args.comp_lrs),
                   pool.submit(worker, "lora", lora, args.lora_lrs)]
        for f in futures:
            results.extend(f.result())
    out = args.run_dir / args.out
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out} ({len(results)} frontier points)")


if __name__ == "__main__":
    main()
