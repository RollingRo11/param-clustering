"""Package the fair sweep into one self-describing JSON.

The raw sweep file is just a list of measurement rows: it carries no baseline,
no units, no statement of what the two arms are or how the points were meant to
be selected. Anyone plotting it from scratch would have to reconstruct all of
that from the training script. This bundles the rows together with the
baselines they are deltas against, the grid that generated them, a field
glossary and the plotting recipe the poster figures use, so the file stands on
its own.

    python3.12 export_fair_sweep.py --out out/fair_sweep_export.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RUN = Path("/dev/shm/geo1b/run1b_streamC4096")

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", type=Path, default=RUN / "lora_fair_sweep.json")
ap.add_argument("--base", type=Path, default=RUN / "fair_sweep_base_ce.json")
ap.add_argument("--out", type=Path,
                default=Path("out/fair_sweep_export.json"))
args = ap.parse_args()

rows = json.loads(args.sweep.read_text())
base = json.loads(args.base.read_text())

UNIFORM = base["_uniform_ce_nats"]                 # ln(128256)
BASE_DE = base["german"]["base_ce_nats"]
CHANCE = UNIFORM - BASE_DE                         # German ΔCE that reaches chance

points = []
for r in rows:
    d = r["eval"]["detail"]
    points.append({
        "objective": r["objective"],
        "arm": r["arm"],
        "budget_german_tokens": r["budget"],
        "lr": r["lr"],
        "lambda_english": r["lam_en"],
        "lambda_romance": r["lam_rom"],
        "step": r["step"],
        "train_seconds": round(r["seconds"], 2),
        # held-out, for reporting
        "eval_delta_ce": {k: d[k] for k in
                          ("german", "english", "fr", "es", "it")},
        # dev/guard, for selecting
        "dev_delta_ce": {k[2:]: v for k, v in r["dev"]["detail"].items()},
        # convenience: the x used by the poster scatter
        "german_vs_chance": round(r["eval"]["german"] - CHANCE, 4),
        "worst_collateral": round(
            max(d[k] for k in ("english", "fr", "es", "it")), 4),
    })

export = {
    "schema": "geo_fair_sweep_export_v1",
    "source": str(args.sweep),
    "n_points": len(points),

    "what_this_is":
        "A hyperparameter sweep comparing two ways of removing German from "
        "Llama-3.2-1B: scaling the weight mass owned by ONE component of a "
        "4096-way attribution-geometry decomposition of the parameters, vs "
        "training a rank-1 LoRA. Every row is one (arm, objective, token "
        "budget, learning rate, lambda, training step) configuration, scored "
        "on German and on four languages it was not asked to remove.",

    "model": {
        "id": "meta-llama/Llama-3.2-1B",
        "vocab_size": 128256,
        "seq_len": 512,
        "eval_precision": "bfloat16 autocast",
    },

    "arms": {
        "component": {
            "label": "single component c3634",
            "trainable_parameters": 112,
            "description":
                "Component 3634 of the C=4096 decomposition owns a share "
                "s_c of every weight entry. The edit is W' = W + (alpha_m - "
                "1) * (s_c * W), one scalar alpha_m per matrix: 16 layers x "
                "7 linear maps = 112 scalars, optimised with Adam. Nothing "
                "else in the model is touched.",
            "learning_rates": [0.1, 0.3],
            "max_steps": 800,
        },
        "lora": {
            "label": "LoRA r=1",
            "trainable_parameters": 704512,
            "description":
                "Rank-1 LoRA adapters on the same 7 linear maps in all 16 "
                "layers, standard init, Adam. 6,290x more trainable "
                "parameters than the component arm.",
            "learning_rates": [1e-4, 3e-4, 1e-3, 3e-3],
            "max_steps": 200,
        },
    },

    "objective": {
        "formula":
            "relu(ln(V) - CE_german) + lambda_en * KL(base_en || edit_en) + "
            "lambda_rom * mean(KL_fr, KL_es, KL_it)",
        "note":
            "The relu term pushes German cross-entropy up to uniform and "
            "stops rewarding further damage past it.",
        "variants": {
            "english_only":
                "lambda_rom = 0. Only English is defended; French, Spanish "
                "and Italian are undefended, so their numbers show what each "
                "method damages incidentally. THIS IS THE INTERESTING ONE.",
            "multilingual":
                "lambda_rom = lambda_en. All four non-target languages are "
                "explicitly protected.",
        },
        "lambda_values": [10.0, 100.0],
    },

    "protocol": {
        "german_token_budgets": [8, 64, 512, 2048],
        "eval_steps": [25, 50, 100, 200, 400, 800],
        "block_split":
            "Points are SELECTED on dev/guard blocks and REPORTED on "
            "strictly held-out blocks. dev_delta_ce and eval_delta_ce come "
            "from disjoint text. Selecting and reporting on the same blocks "
            "would flatter whichever arm has more knobs to tune, which is "
            "LoRA.",
        "why_unequal_step_caps":
            "LoRA converges in ~100 steps and then drifts off its own "
            "objective; the 112-scalar edit needs the full 800. Capping both "
            "at 800 overtrains LoRA into indiscriminate damage, which reads "
            "as 'LoRA is unselective' when it is really 'LoRA was given the "
            "wrong schedule'. Each arm gets the schedule it actually uses, "
            "and the two are compared at MATCHED GERMAN REMOVAL.",
    },

    "units": {
        "all_values": "nats per token",
        "uniform_ce": UNIFORM,
        "base_ce_by_set": {k: v for k, v in base.items()
                           if not k.startswith("_")},
        "german_delta_at_chance": round(CHANCE, 4),
        "reading":
            "eval_delta_ce values are EDITED CE minus UNEDITED CE on the same "
            "text. 0 = untouched. German at +%.3f means German has been "
            "driven all the way to a uniform distribution. Absolute CE = "
            "base_ce_by_set[set] + delta." % CHANCE,
    },

    "fields": {
        "objective": "english_only | multilingual (see objective.variants)",
        "arm": "component | lora",
        "budget_german_tokens": "how many German tokens the edit was fit on",
        "lr": "Adam learning rate",
        "lambda_english": "weight on the English KL preservation term",
        "lambda_romance": "weight on the mean French/Spanish/Italian KL term",
        "step": "training step at which this checkpoint was scored",
        "eval_delta_ce": "held-out ΔCE per language; the numbers to plot",
        "dev_delta_ce": "dev/guard ΔCE per language; the numbers to select on",
        "german_vs_chance": "eval german ΔCE minus german_delta_at_chance; "
                            "0 = exactly at chance, negative = German partly "
                            "survives, positive = pushed past chance",
        "worst_collateral": "max eval ΔCE over the four non-German languages",
        "train_seconds": "wall clock to reach this step",
    },

    "plot_recipe": {
        "poster_figure": "4 languages (rows) x 4 token budgets (columns) "
                         "scatter, objective = english_only",
        "x": "german_vs_chance, limits [-10.1, 15.9], dashed vertical at 0",
        "y": "eval_delta_ce[language]",
        "y_caps": {"english": 1.0, "fr": 1.4, "es": 1.4, "it": 1.4},
        "x_cap": 15.0,
        "offscale":
            "Points past a cap are clamped to the axis edge and drawn as "
            "hollow markers rather than dropped, so the configuration is "
            "still visible but its exact value is not implied.",
        "colors": {"component": "#2a78d6", "lora": "#eb6834"},
        "markers": {"component": "o", "lora": "s"},
        "bar_figure_selection_rule":
            "For the one-point-per-method bar charts: among points with dev "
            "english <= 0.15 AND dev romance <= 0.10, take the one with the "
            "largest dev german. Same rule for both arms. Selecting on "
            "German alone lets a configuration buy its last half-nat by "
            "wrecking Romance, which is what made an earlier version of the "
            "collateral bars erratic.",
    },

    "headline":
        "Under english_only, the component arm reaches chance-level German "
        "removal while leaving the undefended Romance languages nearly "
        "intact; LoRA reaches the same German removal only by also damaging "
        "languages its objective never mentioned. The component edit is "
        "selective without being told what to protect; LoRA is selective "
        "only in proportion to what its objective names.",

    "points": points,
}

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(export, indent=1))
kb = args.out.stat().st_size / 1024
print(f"wrote {args.out} ({len(points)} points, {kb:.0f} KB)")
for obj in ("english_only", "multilingual"):
    for arm in ("component", "lora"):
        sel = [p for p in points if p["objective"] == obj and p["arm"] == arm]
        print(f"  {obj:<13} {arm:<10} {len(sel):3d} points  "
              f"german {min(p['eval_delta_ce']['german'] for p in sel):+.2f} "
              f".. {max(p['eval_delta_ce']['german'] for p in sel):+.2f}")
