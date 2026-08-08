"""Guarded top-k sweep for the German edit inside ONE editor.

Any top-k scaling edit over the ranking's top-16 components is a k=16 alpha
vector with 1s in the unused slots, so a single 16-component editor screens
the whole sweep: per-k tied grids (+ per-coordinate refinement) and, for k=1,
a one-hot grid over every component individually. Selection per k uses the
Romance-guarded rule; each k's winner gets a held-out evaluation.

Grid+refinement screening stands in for per-k gradient training; on this
protocol the constrained optima have always been found by the screen (the
gradient phase only matters for the deep-erasure lambda configs, which the
guard rejects anyway). Trained vectors from an existing k-run result are
folded into the candidate pool when present.
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
from german_vpd_guard import ONE_HOT_GRID, eligible, pick, screen

TIED_GRID = [-20.0, -15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0,
             3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0,
             20.0]


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
    parser.add_argument("--k_values", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--en_budget", type=float, default=0.1)
    parser.add_argument("--romance_guards", type=float, nargs="+",
                        default=[1.0, 0.5, 0.3])
    parser.add_argument("--prior_result",
                        default="german_vpd_prop1b_k8.json",
                        help="existing run whose candidates are folded in")
    parser.add_argument("--sweep_rows", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh_data", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.artifact_root / args.tag
    args.bank_path = args.run_dir / f"banks_{args.banks_tag}.pt"
    args.data_cache = args.run_dir / "german_vpd_data.pt"
    torch.manual_seed(args.seed)

    ranking = json.loads(
        (args.run_dir / "german_vpd_ranking.json").read_text())
    components = [row["component"]
                  for row in ranking["inspected_candidates"]
                  [:args.total_components]]
    total = len(components)
    log(f"top-{total} components by German specificity: {components}")

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

    candidates = []
    seen = set()

    def add(alpha, origin):
        key = tuple(round(a, 4) for a in alpha)
        if key not in seen:
            seen.add(key)
            candidates.append({"alpha": [float(a) for a in alpha],
                               "source": origin})

    # k=1: every component individually.
    for slot in range(total):
        for value in ONE_HOT_GRID:
            vector = [1.0] * total
            vector[slot] = value
            add(vector, f"k1:c{components[slot]}")
    # k>=2: tied over the top-k prefix.
    for k in args.k_values:
        if k < 2 or k > total:
            continue
        for value in TIED_GRID:
            vector = [value] * k + [1.0] * (total - k)
            add(vector, f"k{k}:tied")
    # Fold in an existing run's trained/screened vectors, padded to k=16.
    prior_path = args.run_dir / args.prior_result
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        prior_components = prior["components"]
        index = {c: components.index(c) for c in prior_components
                 if c in components}
        if len(index) == len(prior_components):
            for row in prior["training"]["dev_candidates"]:
                vector = [1.0] * total
                for c, a in zip(prior_components, row["alpha"]):
                    vector[index[c]] = float(a)
                add(vector, f"prior:{row['source']}")
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

    def slots_of(row):
        return [i for i, a in enumerate(row["alpha"])
                if round(a, 4) != 1.0]

    primary = args.romance_guards[0]
    per_k = {}
    for k in args.k_values:
        pool = [row for row in candidates if len(slots_of(row)) <= k
                and all(slot < k or row["source"].startswith("k1:")
                        for slot in slots_of(row))]
        # k=1 pool: any single-slot vector; k>=2 pool: vectors confined to
        # the top-k prefix (tied/prior/refined) or single-slot vectors.
        if k == 1:
            pool = [row for row in candidates if len(slots_of(row)) <= 1]
        best = pick(pool, args.en_budget, primary)
        if best is not None:
            refined = []
            for vector in refinement_vectors(best["alpha"]):
                key = tuple(round(a, 4) for a in vector)
                if key not in seen:
                    seen.add(key)
                    refined.append({"alpha": [float(a) for a in vector],
                                    "source": f"k{k}:refine"})
            if refined:
                screen(editors, guard_blocks, refined, args.sweep_rows)
                candidates.extend(refined)
                pool.extend(refined)
        per_k[k] = {
            str(guard): pick(pool, args.en_budget, guard)
            for guard in args.romance_guards}
        for guard in args.romance_guards:
            row = per_k[k][str(guard)]
            if row is None:
                log(f"k={k} guard<{guard:g}: none eligible")
            else:
                romance = max(row[f"{lang}_guard"]["delta_ce"]
                              for lang in ("fr", "es", "it"))
                log(f"k={k} guard<{guard:g}: {fmt_vec(row['alpha'])} "
                    f"({row['source']}) de={row['de_dev']['delta_ce']:+.3f} "
                    f"en={row['en_dev']['delta_ce']:+.3f} "
                    f"romance={romance:+.3f}")
    log(f"screening done ({time.perf_counter() - t0:.0f}s)")

    # Held-out evaluation of each k's primary-guard winner.
    datasets = {
        "german_europarl": data["de_eval"],
        "english_pile": data["pile_en_eval"],
        "english_europarl": data["en_europarl_eval"],
        "codeparrot": data["code_eval"],
        "french_europarl_heldout": data["fr_eval"][2:],
        "spanish_europarl_heldout": data["es_eval"][2:],
        "italian_europarl_heldout": data["it_eval"][2:],
    }
    winners = {k: per_k[k][str(primary)] for k in args.k_values
               if per_k[k][str(primary)] is not None}

    def evaluate_vector(worker_editor, alpha):
        rows = {}
        for name, idx in datasets.items():
            original = base_logits(worker_editor, [idx])[0]
            rows[name] = forward_metrics(worker_editor, idx, original, alpha)
        return rows

    evaluations = {}
    items = list(winners.items())
    with ThreadPoolExecutor(len(editors)) as pool_exec:
        queues = [items[i::len(editors)] for i in range(len(editors))]

        def work(worker_editor, queue):
            return {k: evaluate_vector(worker_editor, row["alpha"])
                    for k, row in queue}

        futures = [pool_exec.submit(work, worker, queue)
                   for worker, queue in zip(editors, queues) if queue]
        for future in futures:
            evaluations.update(future.result())
    for k in sorted(evaluations):
        rows = evaluations[k]
        log(f"EVAL k={k} {fmt_vec(winners[k]['alpha'])}: "
            f"de={rows['german_europarl']['delta_ce']:+.3f} "
            f"en_pile={rows['english_pile']['delta_ce']:+.3f} "
            f"fr/es/it="
            f"{rows['french_europarl_heldout']['delta_ce']:+.3f}/"
            f"{rows['spanish_europarl_heldout']['delta_ce']:+.3f}/"
            f"{rows['italian_europarl_heldout']['delta_ce']:+.3f}")

    best_k = max(evaluations, key=lambda k:
                 evaluations[k]["german_europarl"]["delta_ce"])
    verification = editor.verify_in_place(
        data["de_eval"][:1, :32].to(devices[0]), winners[best_k]["alpha"])
    rollouts = greedy_rollouts(
        editors, tokenizer, winners[best_k]["alpha"], args.max_new_tokens)

    output = args.run_dir / "german_topk_sweep.json"
    output.write_text(json.dumps({
        "format": "german_topk_sweep_v1",
        "components": components,
        "k_values": args.k_values,
        "en_budget": args.en_budget,
        "romance_guards": args.romance_guards,
        "per_k_selections": {str(k): value for k, value in per_k.items()},
        "evaluations": {str(k): value for k, value in evaluations.items()},
        "best_k": best_k,
        "best_alpha": winners[best_k]["alpha"],
        "literal_weight_edit_verification": verification,
        "rollouts": rollouts,
        "candidates": candidates,
    }, indent=2))
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
