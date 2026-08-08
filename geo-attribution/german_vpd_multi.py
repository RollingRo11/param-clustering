"""Multilingual-preserve scalar fine-tune of German components.

The English-only KL preserve term lets destructive configs ride shared
European-language machinery (the post itself flags this limitation), and
selection-side Romance guards can only REJECT candidates — they cannot
construct cancellation. This runner puts the preservation of English AND
French/Spanish/Italian directly into the gradient objective:

    relu(log V - German CE)
      + lambda_en  * KL(base_en || edit_en)
      + lambda_rom * mean_l KL(base_l || edit_l),  l in {fr, es, it}

over k signed per-component scalars, so the optimizer can trade components
against each other to cancel Romance damage while keeping German damage.
Warm start scores a candidate pool under the SAME multilingual objective.
Selection and final evaluation reuse the guarded rule and held-out splits.
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
from german_vpd_1b import (ComponentEditor, alpha_sweep, base_logits, ce_each,
                           fmt_vec, forward_metrics, greedy_rollouts, log,
                           prepare_data, refinement_vectors)
from german_vpd_guard import ONE_HOT_GRID, pick, screen

ROMANCE = ("fr", "es", "it")
TIED_GRID = [-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
             8.0, 10.0, 12.0, 15.0, 20.0]


def train_blocks(data):
    """Language -> training blocks. Romance training uses rank + first eval
    block; eval block 1 stays a selection guard and blocks 2+ stay held out."""
    blocks = {"de": data["de_train"], "en": data["en_train"]}
    for lang in ROMANCE:
        blocks[lang] = torch.cat(
            [data[f"{lang}_rank"], data[f"{lang}_eval"][:1]])
    return blocks


def multilingual_objective(row, ceiling, lam_en, lam_rom):
    destroy = max(0.0, ceiling - row["de"]["ce"])
    romance = sum(row[lang]["kl"] for lang in ROMANCE) / len(ROMANCE)
    return destroy + lam_en * row["en"]["kl"] + lam_rom * romance


def warm_start(args, editors, data, components, ceiling):
    """Score a candidate pool under the multilingual objective per config."""
    total = len(components)
    pool = []
    seen = set()

    def add(alpha, origin):
        key = tuple(round(a, 4) for a in alpha)
        if key not in seen:
            seen.add(key)
            pool.append({"alpha": [float(a) for a in alpha],
                         "source": origin})

    for slot in range(total):
        for value in ONE_HOT_GRID:
            vector = [1.0] * total
            vector[slot] = value
            add(vector, f"one_hot:c{components[slot]}")
    for value in TIED_GRID:
        add([value] * total, "tied")
    sweep_path = args.run_dir / "german_topk_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        if sweep["components"] == components:
            ranked = sorted(
                (row for row in sweep["candidates"] if "de_dev" in row),
                key=lambda row: -row["de_dev"]["delta_ce"])
            for row in ranked[:80]:
                add(row["alpha"], f"sweep:{row['source']}")
    log(f"warm-start pool: {len(pool)} vectors")

    blocks = train_blocks(data)
    vectors = [row["alpha"] for row in pool]
    for lang, idx_blocks in blocks.items():
        sweep_out = alpha_sweep(editors, idx_blocks, vectors, args.sweep_rows)
        for i, row in enumerate(pool):
            row[lang] = {"ce": sweep_out["ce"][i], "kl": sweep_out["kl"][i]}
    inits = []
    for lam_en, lam_rom in args.lambda_pairs:
        best = min(pool, key=lambda row: multilingual_objective(
            row, ceiling, lam_en, lam_rom))
        inits.append(best["alpha"])
        log(f"warm init (l_en={lam_en:g}, l_rom={lam_rom:g}): "
            f"{fmt_vec(best['alpha'])} [{best['source']}] "
            f"deCE={best['de']['ce']:.2f} enKL={best['en']['kl']:.3f} "
            f"romKL={sum(best[l]['kl'] for l in ROMANCE)/3:.3f}")
    return inits, pool


def _train_worker(editor, alphas, lam_en, lam_rom, idx_rows, base_logp,
                  ceiling):
    """Grad-accumulated fwd/bwd per config across all 5 languages.

    Configs run sequentially: the per-row-alpha path saves every per-slot
    component output for autograd, so at k=16 a fully batched group would
    hold ~17x activations and OOM a B200.
    """
    device = editor.device
    configs = alphas.shape[0]
    idx = idx_rows.to(device)                      # (5, T): de en fr es it
    german_out, en_out, rom_out, loss_out = [], [], [], []
    for i in range(configs):
        alpha_batch = alphas[i:i + 1].expand(5, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = editor.logits(idx, alpha_batch)
            german_ce = ce_each(logits[0:1], idx[0:1]).squeeze(0)
            kls = []
            for j in range(1, 5):
                edited_logp = F.log_softmax(
                    logits[j:j + 1, :-1].float(), -1)
                kls.append(F.kl_div(
                    edited_logp, base_logp[j - 1], log_target=True,
                    reduction="none").sum(-1).mean())
            destroy = F.relu(ceiling - german_ce)
            romance = (kls[1] + kls[2] + kls[3]) / 3.0
            loss = destroy + lam_en[i] * kls[0] + lam_rom[i] * romance
        loss.backward()
        german_out.append(german_ce.detach())
        en_out.append(kls[0].detach())
        rom_out.append(romance.detach())
        loss_out.append(loss.detach())
        del logits
    return (torch.stack(german_out).cpu(), torch.stack(en_out).cpu(),
            torch.stack(rom_out).cpu(), torch.stack(loss_out).cpu())


def train(args, editors, data, components, ceiling, inits):
    configs = len(args.lambda_pairs)
    blocks = train_blocks(data)
    share = math.ceil(configs / len(editors))
    groups = []
    for slot, editor in enumerate(editors):
        lo, hi = slot * share, min((slot + 1) * share, configs)
        if lo >= hi:
            continue
        lam = args.lambda_pairs[lo:hi]
        groups.append({
            "editor": editor,
            "alphas": torch.nn.Parameter(torch.tensor(
                inits[lo:hi], dtype=torch.float32, device=editor.device)),
            "lam_en": torch.tensor([p[0] for p in lam], device=editor.device),
            "lam_rom": torch.tensor([p[1] for p in lam],
                                    device=editor.device),
        })
    optimizer = torch.optim.Adam(
        [group["alphas"] for group in groups], lr=args.lr)

    # Per-device base log-probs for every preserve block.
    for group in groups:
        editor = group["editor"]
        group["base_logp"] = {}
        for lang in ("en",) + ROMANCE:
            rows = []
            for block in blocks[lang]:
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=True):
                    base = editor.logits(block[None].to(editor.device), None)
                rows.append(F.log_softmax(base[:, :-1].float(), -1))
                del base
            group["base_logp"][lang] = rows

    history = []
    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(len(groups))
    counts = {lang: len(blocks[lang]) for lang in blocks}
    for step in range(args.steps):
        chosen = {lang: (step * (7 if lang == "en" else 3)) % counts[lang]
                  for lang in blocks}
        chosen["de"] = step % counts["de"]
        idx_rows = torch.stack(
            [blocks[lang][chosen[lang]]
             for lang in ("de", "en", "fr", "es", "it")])
        futures = []
        optimizer.zero_grad(set_to_none=True)
        for group in groups:
            base_logp = [group["base_logp"][lang][chosen[lang]]
                         for lang in ("en",) + ROMANCE]
            futures.append(pool.submit(
                _train_worker, group["editor"], group["alphas"],
                group["lam_en"], group["lam_rom"], idx_rows, base_logp,
                ceiling))
        outputs = [future.result() for future in futures]
        optimizer.step()
        with torch.no_grad():
            for group in groups:
                group["alphas"].nan_to_num_(
                    nan=1.0, posinf=args.alpha_max, neginf=args.alpha_min)
                group["alphas"].clamp_(args.alpha_min, args.alpha_max)
        if step % args.log_every == 0 or step + 1 == args.steps:
            german = torch.cat([out[0] for out in outputs])
            en_kl = torch.cat([out[1] for out in outputs])
            rom_kl = torch.cat([out[2] for out in outputs])
            history.append({
                "step": step,
                "alpha": torch.cat([g["alphas"].detach().cpu()
                                    for g in groups]).tolist(),
                "german_ce": german.tolist(),
                "en_kl": en_kl.tolist(),
                "rom_kl": rom_kl.tolist(),
            })
            log(f"step {step:04d}: " + ", ".join(
                f"(l_en={pair[0]:g},l_rom={pair[1]:g}) deCE={ce:.2f} "
                f"enKL={ekl:.3f} romKL={rkl:.3f}"
                for pair, ce, ekl, rkl in zip(
                    args.lambda_pairs, german.tolist(), en_kl.tolist(),
                    rom_kl.tolist())))
    pool.shutdown()
    trained = torch.cat(
        [group["alphas"].detach().cpu() for group in groups]).tolist()
    log(f"training done ({time.perf_counter() - t0:.0f}s)")
    return trained, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--total_components", type=int, default=16)
    parser.add_argument("--lambda_pairs", type=float, nargs="+",
                        default=[3.0, 3.0, 10.0, 10.0, 10.0, 30.0,
                                 30.0, 30.0, 30.0, 100.0],
                        help="flat (lambda_en, lambda_rom) pairs")
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--en_budget", type=float, default=0.1)
    parser.add_argument("--romance_guards", type=float, nargs="+",
                        default=[1.0, 0.5])
    parser.add_argument("--alpha_min", type=float, default=-50.0)
    parser.add_argument("--alpha_max", type=float, default=100.0)
    parser.add_argument("--sweep_rows", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    args.lambda_pairs = [
        (args.lambda_pairs[i], args.lambda_pairs[i + 1])
        for i in range(0, len(args.lambda_pairs), 2)]
    torch.manual_seed(args.seed)

    ranking = json.loads(
        (args.run_dir / "german_vpd_ranking.json").read_text())
    components = [row["component"]
                  for row in ranking["inspected_candidates"]
                  [:args.total_components]]
    log(f"multilingual scalar fine-tune over {components}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        geo1b.MODEL_ID, revision=geo1b.MODEL_REVISION)
    data = prepare_data(args, tokenizer)
    devices = args.devices or [
        f"cuda:{i}" for i in range(torch.cuda.device_count())]
    bank = torch.load(args.bank_path, weights_only=True, map_location="cpu",
                      mmap=True)
    target = geo1b.load_target_1b(devices[0])
    editor = ComponentEditor(target, bank, components, devices[0])
    del bank
    editors = [editor] + [editor.replicate(dev) for dev in devices[1:]]
    ceiling = math.log(editor.target.hf.config.vocab_size)

    inits, pool = warm_start(args, editors, data, components, ceiling)
    trained, history = train(
        args, editors, data, components, ceiling, inits)

    # Guarded selection over trained vectors + the warm pool + refinement.
    candidates = [{"alpha": vector,
                   "source": f"trained:l_en={pair[0]:g},l_rom={pair[1]:g}"}
                  for vector, pair in zip(trained, args.lambda_pairs)]
    seen = {tuple(round(a, 4) for a in row["alpha"]) for row in candidates}
    for row in pool:
        key = tuple(round(a, 4) for a in row["alpha"])
        if key not in seen:
            seen.add(key)
            candidates.append(
                {"alpha": row["alpha"], "source": row["source"]})
    guard_blocks = {
        "de_dev": data["de_dev"],
        "en_dev": data["en_dev"],
        "fr_guard": data["fr_eval"][1:2],
        "es_guard": data["es_eval"][1:2],
        "it_guard": data["it_eval"][1:2],
    }
    screen(editors, guard_blocks, candidates, args.sweep_rows)
    primary = args.romance_guards[0]
    best = pick(candidates, args.en_budget, primary)
    if best is not None:
        refined = []
        for vector in refinement_vectors(best["alpha"]):
            key = tuple(round(a, 4) for a in vector)
            if key not in seen:
                seen.add(key)
                refined.append({"alpha": [float(a) for a in vector],
                                "source": "guard_refine"})
        if refined:
            screen(editors, guard_blocks, refined, args.sweep_rows)
            candidates.extend(refined)
    selections = {}
    for guard in args.romance_guards:
        row = pick(candidates, args.en_budget, guard)
        selections[str(guard)] = row
        if row is not None:
            romance = max(row[f"{lang}_guard"]["delta_ce"]
                          for lang in ROMANCE)
            log(f"guard<{guard:g}: {fmt_vec(row['alpha'])} ({row['source']}) "
                f"dev de={row['de_dev']['delta_ce']:+.3f} "
                f"en={row['en_dev']['delta_ce']:+.3f} romance={romance:+.3f}")
        else:
            log(f"guard<{guard:g}: none eligible")
    selected = selections[str(primary)]
    if selected is None:
        raise SystemExit("no candidate met the primary guard")
    alpha = selected["alpha"]

    datasets = {
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "codeparrot": data["code_eval"],
        "french_europarl_heldout": data["fr_eval"][2:],
        "spanish_europarl_heldout": data["es_eval"][2:],
        "italian_europarl_heldout": data["it_eval"][2:],
    }
    queues = [[] for _ in editors]
    for i, item in enumerate(datasets.items()):
        queues[i % len(editors)].append(item)

    def work(worker_editor, queue):
        rows = {}
        for name, idx in queue:
            original = base_logits(worker_editor, [idx])[0]
            rows[name] = forward_metrics(worker_editor, idx, original, alpha)
        return rows

    evaluation = {}
    with ThreadPoolExecutor(len(editors)) as pool_exec:
        futures = [pool_exec.submit(work, worker, queue)
                   for worker, queue in zip(editors, queues) if queue]
        for future in futures:
            evaluation.update(future.result())
    for name in datasets:
        row = evaluation[name]
        log(f"eval {name}: CE {row['base_ce']:.3f} -> {row['edited_ce']:.3f} "
            f"({row['delta_ce']:+.3f})")
    verification = editor.verify_in_place(
        data["de_eval"][:1, :32].to(devices[0]), alpha)
    log("literal weight-edit equivalence: "
        f"max |delta logit|={verification['max_abs_logit_error']:.3e}")
    rollouts = greedy_rollouts(editors, tokenizer, alpha, args.max_new_tokens)

    output = args.run_dir / "german_vpd_multilingual.json"
    output.write_text(json.dumps({
        "format": "german_vpd_multilingual_v1",
        "objective": ("relu(logV-deCE) + lam_en*KL_en "
                      "+ lam_rom*mean(KL_fr,KL_es,KL_it)"),
        "components": components,
        "lambda_pairs": args.lambda_pairs,
        "steps": args.steps,
        "lr": args.lr,
        "trained": trained,
        "history": history,
        "selections": selections,
        "selected": selected,
        "alpha": alpha,
        "evaluation": evaluation,
        "literal_weight_edit_verification": verification,
        "rollouts": rollouts,
        "candidates": candidates,
    }, indent=2))
    torch.save({
        "format": "softpart_component_scalar_adapter_v2",
        "model": geo1b.model_identity(),
        "bank": str(args.bank_path),
        "components": components,
        "alpha": alpha,
        "formula": "W' = W + sum_i (alpha_i - 1) * W_component_i",
        "result": str(output),
    }, args.run_dir / "german_vpd_multilingual_adapter.pt")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
