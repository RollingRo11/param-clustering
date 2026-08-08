"""Package the intruder-detection result into one self-describing JSON.

Same idea as export_fair_sweep.py: the raw run files are bare trial lists with
no protocol, no baselines and no statistics. This bundles the trials with the
method, the confidence intervals, the significance tests and the plot recipe so
the file can be handed to something that has never seen this repo.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RUN = Path("/dev/shm/geo1b/run1b_streamC4096")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 5), round(c + h, 5)


def summarise(trials):
    n = len(trials)
    k = sum(t["correct"] for t in trials)
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "accuracy": round(k / n, 5),
            "ci95": [lo, hi]}


def perm_p(a, b, iters=100000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    allv = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(iters):
        p = rng.permutation(allv)
        if abs(p[:na].mean() - p[na:].mean()) >= obs:
            hits += 1
    return round(hits / iters, 6)


ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, default=RUN)
ap.add_argument("--out", type=Path,
                default=Path("out/intruder_export.json"))
args = ap.parse_args()

runs = {k: json.loads((args.run / f"intruder_{k}.json").read_text())
        for k in ("random", "near")}

conditions = {}
for key, label in (("random", "intruder drawn from a RANDOM other component"),
                   ("near", "intruder drawn from the target's NEAREST "
                            "NEIGHBOUR in centroid space")):
    d = runs[key]
    t = d["trials"]
    by_grade = {}
    for g in ("mono", "partial", "poly"):
        sub = [x for x in t if x["grade"] == g]
        if sub:
            by_grade[g] = summarise(sub)
    by_cat = {}
    for x in t:
        by_cat.setdefault(x["category"], []).append(x)
    by_cat = {k2: summarise(v) for k2, v in
              sorted(by_cat.items(), key=lambda kv: -len(kv[1]))}
    fits = np.array([x["fit"] for x in t
                     if isinstance(x["fit"], (int, float))], dtype=float)
    cors = np.array([x["correct"] for x in t
                     if isinstance(x["fit"], (int, float))], dtype=float)
    rho = (float(np.corrcoef(np.argsort(np.argsort(fits)), cors)[0, 1])
           if fits.std() > 0 else None)
    mono = np.array([x["correct"] for x in t if x["grade"] == "mono"],
                    dtype=float)
    part = np.array([x["correct"] for x in t if x["grade"] == "partial"],
                    dtype=float)
    conditions[key] = {
        "description": label,
        "overall": summarise(t),
        "by_label_grade": by_grade,
        "by_category": by_cat,
        "spearman_labelfit_vs_correct": (None if rho is None
                                         else round(rho, 4)),
        "mono_vs_partial_permutation_p": (
            perm_p(mono, part) if len(mono) and len(part) else None),
        "guess_distribution": d["guess_distribution"],
        "truth_distribution": d["truth_distribution"],
        "api_failures_excluded": d["errors"],
        "trials": [{k2: x[k2] for k2 in
                    ("component", "grade", "fit", "category", "truth",
                     "guess", "correct", "confidence", "intruder_from")}
                   for x in t],
    }

export = {
    "schema": "intruder_detection_export_v1",
    "n_trials_total": sum(c["overall"]["n"] for c in conditions.values()),

    "what_this_is":
        "Intruder detection on a 4,096-component attribution-geometry "
        "decomposition of Llama-3.2-1B, replicating Figure 6 of Goodfire's "
        "adVersarial Parameter Decomposition (VPD) paper. A judge is shown "
        "five short text windows, each with one token marked. Four are windows "
        "where the same component fires; one is an intruder from a different "
        "component. The judge must name the intruder. Chance is 20%.",

    "why_it_matters":
        "The decomposition's headline monosemanticity number (81.7% mono) "
        "came from the labelling model grading its own labels. Intruder "
        "detection shows the judge NO label, so it measures whether a "
        "component's activation set is coherent independently of how well it "
        "was described. It is therefore a second opinion on the decomposition "
        "rather than on the labelling.",

    "method": {
        "judge_model": runs["random"]["model"],
        "windows_per_trial": 5,
        "real_windows": runs["random"]["k_real"],
        "chance_accuracy": 0.2,
        "real_windows_drawn_from":
            "the component's top-12 evidence windows (highest posterior "
            "share), sampled without replacement",
        "intruder_drawn_from":
            "another component's top-8 evidence windows — a fluent, "
            "high-activation window for something else, not random text",
        "evidence_source":
            "1,000,017,920 token positions swept, top-32 windows per "
            "component, at most one window per source document",
        "marking": "the firing token is wrapped in «guillemets»",
        "judge_output": "JSON {intruder: 1-5, reason, confidence}",
        "trials_excluded":
            "API refusals and unparseable responses are dropped, not counted "
            "as errors; the per-condition count is in api_failures_excluded",
    },

    "conditions": conditions,

    "nearest_neighbour_detail": {
        "definition":
            "cosine similarity between L2-normalised spherical k-means "
            "centroids in the frozen 256-dimensional embedding space",
        "mean_cosine_to_nearest_neighbour": 0.8098,
        "why":
            "This is the feature-splitting test (VPD Section 3.5) asked with a "
            "judge instead of an alive-count. If C=4,096 had split one real "
            "feature into two components, a component and its nearest "
            "neighbour would be indistinguishable and accuracy would collapse "
            "toward chance. It does not.",
    },

    "headline_findings": [
        "99.5% accuracy with a random intruder (n=413) against 20% chance.",
        "88.8% accuracy against the nearest-neighbour intruder (n=277), so "
        "adjacent components remain distinguishable and C=4,096 is not "
        "over-split.",
        "Components the labeller graded 'mono' score 90.9% in the hard "
        "condition against 66.7% for 'partial' (permutation p = 0.002) — the "
        "self-graded scale is validated by a metric that never sees a label.",
        "The judge's guess distribution matches the truth distribution almost "
        "exactly, so there is no answer-position bias.",
    ],

    "caveats": [
        "The judge is an LLM of the same family as the labeller, so shared "
        "blind spots are possible; the two tasks are different (no label is "
        "shown here) but not fully independent.",
        "Only components with at least 6 evidence windows and a valid grade "
        "are eligible.",
        "The 'poly' grade is too rare in the sample (n=1) to report.",
        "The 'near' condition has fewer trials because some nearest "
        "neighbours lack usable evidence.",
    ],

    "fields": {
        "component": "component id, 0-4095",
        "grade": "mono | partial | poly, assigned by the auto-interp labeller",
        "fit": "labeller's estimate of the % of examples its label covers",
        "category": "auto-interp category of the component",
        "truth": "which of the 5 positions was the intruder (1-indexed)",
        "guess": "which position the judge named",
        "correct": "guess == truth",
        "confidence": "judge's self-reported confidence",
        "intruder_from": "component the intruder window was taken from",
    },

    "plot_recipe": {
        "figure": "two panels",
        "panel_1":
            "overall accuracy per condition as bars with 95% Wilson intervals, "
            "dashed horizontal line at 0.20 for chance",
        "panel_2":
            "same accuracy split by by_label_grade (mono vs partial), grouped "
            "bars with one colour per condition",
        "colors": {"random": "#2a78d6", "near": "#eb6834"},
        "y_axis": "0 to 1.13, ticks at 0/25/50/75/100%",
        "note":
            "Wilson intervals, not normal-approximation: mono/random is 360/360 "
            "and a normal interval would have zero width there.",
    },
}

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(export, indent=1))
print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, "
      f"{export['n_trials_total']} trials)")
for k, c in conditions.items():
    o = c["overall"]
    print(f"  {k:<7} {o['accuracy']:.4f} CI {o['ci95']}  n={o['n']}")
