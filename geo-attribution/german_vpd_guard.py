"""Romance-guarded reselection over an existing german_vpd k-component run.

The English-only dev constraint cannot distinguish a German component from a
generic European-languages component (c1014@+9 destroyed fr/es/it to chance
while passing the English budget). This runner reuses the completed run's
trained + screened alpha vectors — no retraining — and reselects under the
post's rule extended with a Romance guard:

    maximize dev German CE
    s.t. dev English dCE < en_budget
         max(fr, es, it) guard dCE < romance_guard

Guard blocks are the first two fr/es/it eval blocks; the final evaluation
uses only the held-back remainder for those languages.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import torch

import geo1b  # noqa: F401
from german_vpd_1b import (ComponentEditor, alpha_sweep, base_logits,
                           fmt_vec, forward_metrics, greedy_rollouts, log,
                           prepare_data, refinement_vectors)

ONE_HOT_GRID = [-20.0, -15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0,
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]


def screen(editors, blocks_by_name, candidates, sweep_rows):
    vectors = [row["alpha"] for row in candidates]
    for name, blocks in blocks_by_name.items():
        sweep = alpha_sweep(editors, blocks, vectors, sweep_rows)
        for i, row in enumerate(candidates):
            row[name] = {"base_ce": sweep["base_ce"],
                         "edited_ce": sweep["ce"][i],
                         "delta_ce": sweep["ce"][i] - sweep["base_ce"],
                         "kl_from_base": sweep["kl"][i]}
    return candidates


def eligible(row, en_budget, guard):
    return (row["en_dev"]["delta_ce"] < en_budget
            and max(row[f"{lang}_guard"]["delta_ce"]
                    for lang in ("fr", "es", "it")) < guard)


def pick(candidates, en_budget, guard):
    rows = [row for row in candidates if eligible(row, en_budget, guard)]
    if not rows:
        return None
    return max(rows, key=lambda row: row["de_dev"]["edited_ce"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_stream")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--result_name", default="german_vpd_prop1b_k6.json")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--train_tokens", type=int, default=2048)
    parser.add_argument("--eval_blocks", type=int, default=4)
    parser.add_argument("--en_budget", type=float, default=0.1)
    parser.add_argument("--romance_guards", type=float, nargs="+",
                        default=[1.0, 0.5, 0.3])
    parser.add_argument("--sweep_rows", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    torch.manual_seed(args.seed)

    source = json.loads((args.run_dir / args.result_name).read_text())
    components = source["components"]
    k = len(components)
    log(f"guarded reselection over components {components}")

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

    # Candidate set: everything the completed run trained or screened, plus a
    # per-component one-hot grid so every component's own frontier is visible
    # to the guarded rule.
    candidates = []
    seen = set()

    def add(alpha, origin):
        key = tuple(round(a, 4) for a in alpha)
        if key not in seen:
            seen.add(key)
            candidates.append({"alpha": [float(a) for a in alpha],
                               "source": origin})

    for row in source["training"]["dev_candidates"]:
        add(row["alpha"], f"prior:{row['source']}")
    for slot in range(k):
        for value in ONE_HOT_GRID:
            vector = [1.0] * k
            vector[slot] = value
            add(vector, f"one_hot:c{components[slot]}")
    log(f"screening {len(candidates)} candidates")

    guard_blocks = {
        "de_dev": data["de_dev"],
        "en_dev": data["en_dev"],
        "fr_guard": data["fr_eval"][:2],
        "es_guard": data["es_eval"][:2],
        "it_guard": data["it_eval"][:2],
    }
    t0 = time.perf_counter()
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
    log(f"screening done ({time.perf_counter() - t0:.0f}s)")

    selections = {}
    for guard in args.romance_guards:
        row = pick(candidates, args.en_budget, guard)
        selections[str(guard)] = row
        if row is None:
            log(f"guard<{guard:g}: no eligible candidate")
        else:
            romance = max(row[f"{lang}_guard"]["delta_ce"]
                          for lang in ("fr", "es", "it"))
            log(f"guard<{guard:g}: {fmt_vec(row['alpha'])} ({row['source']}) "
                f"dev de={row['de_dev']['delta_ce']:+.3f} "
                f"en={row['en_dev']['delta_ce']:+.3f} "
                f"romance_max={romance:+.3f}")

    selected = selections[str(primary)]
    if selected is None:
        raise SystemExit("no candidate satisfied the primary guard")
    alpha = selected["alpha"]

    # Held-out evaluation: full de/en/pile/code sets; the fr/es/it blocks the
    # guard never saw.
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
    with ThreadPoolExecutor(len(editors)) as pool:
        futures = [pool.submit(work, worker, queue)
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

    def strip(row):
        return row if row is None else dict(row)

    output = args.run_dir / args.result_name.replace(".json", "_guarded.json")
    output.write_text(json.dumps({
        "format": "german_vpd_guarded_reselection_v1",
        "source_result": str(args.run_dir / args.result_name),
        "components": components,
        "rule": ("max dev German CE s.t. dev English dCE < en_budget and "
                 "max(fr,es,it) guard dCE < romance_guard"),
        "en_budget": args.en_budget,
        "romance_guards": args.romance_guards,
        "selections": {key: strip(value)
                       for key, value in selections.items()},
        "selected": strip(selected),
        "alpha": alpha,
        "candidates": candidates,
        "evaluation": evaluation,
        "literal_weight_edit_verification": verification,
        "rollouts": rollouts,
    }, indent=2))
    torch.save({
        "format": "softpart_component_scalar_adapter_v2",
        "model": geo1b.model_identity(),
        "bank": str(args.bank_path),
        "components": components,
        "alpha": alpha,
        "formula": "W' = W + sum_i (alpha_i - 1) * W_component_i",
        "result": str(output),
    }, args.run_dir / args.result_name.replace(".json", "_guarded_adapter.pt"))
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
