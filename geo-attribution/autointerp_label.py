"""Label stage for the streaming-fingerprint evidence (all C components).

autointerp67's label stage predates the C=4096 stream artifacts and points at a
different .env / evidence layout. This one consumes
`<run_dir>/evidence_<banks_tag>.json` (written by autointerp_stream.py) and adds
the monosemanticity grade the manual survey produced by hand, so the same
judgement is applied uniformly to every component instead of an 80-component
sample.

  python3.12 autointerp_label.py --tag run1b_streamC4096 --banks_tag prop1b

Resumable: re-running skips components already in the catalog (and retries ones
whose previous attempt errored).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geo1b
from german_vpd_1b import log

SYSTEM = """You label components of a decomposed language model (Llama-3.2-1B \
decomposed into 4096 cross-layer parameter components). Each component is shown \
with context windows from real text where its attribution fingerprint was most \
active; the token at which it fired is marked with «guillemets». Every example \
comes from a DIFFERENT source document, so a shared pattern is a real \
generalization, not memorization of one document. The model predicts the NEXT \
token, so a component may respond to the current token, the surrounding \
context, or what is about to come.

Judge strictly. Most components in this decomposition are narrow: a specific \
notation slot (the token after `{` in CSS, the `="` in markup), a specific \
subword-continuation class (surname continuations, toponym continuations, \
acronym openers), or a register-conditioned function-word slot. "Tokens in \
English text" or "punctuation" is NOT a pattern — if that is the best you can \
say, the component is polysemantic.

Reply with ONLY a JSON object:
{"label": "<concise functional description, <=12 words>",
 "category": "<one of: current-token, next-token-prediction, syntax, \
semantic-topic, boundary, formatting-notation, subword-continuation, other>",
 "fit": <integer 0-100: percent of the shown examples your label actually \
covers — count them>,
 "mono": "<one of: mono, partial, poly>",
 "confidence": "<high|medium|low>"}

Grade from `fit`, applied to the SPECIFIC label you wrote:
  mono    = fit >= 85. Nearly every example is an instance of your label.
  partial = fit 55-84, or the only label covering them all had to be broad.
  poly    = fit < 55; no specific label covers a majority. Then label it \
"polysemantic/unclear".

Calibration — do not over-penalize:
- A narrow FAMILY is one specific pattern. "Surname subword continuations", \
"CSS property slot after `{`", "preposition before a biomedical noun phrase" \
each count as mono when the examples fit, even though the surface tokens all \
differ. Varying tokens is what generalization looks like, not incoherence.
- Different domains (medicine, law, code) sharing one structural role is still \
mono if the role is what unifies them.
- Reserve partial for a real drifting tail (a sharp head, then examples your \
label does not cover) or for a genuinely broad label.
- A couple of unclear examples out of twenty do not make a component partial."""


def grade_from_fit(fit):
    """The rubric's own thresholds, applied to a reported fit percentage."""
    if not isinstance(fit, (int, float)) or isinstance(fit, bool):
        return "unknown"
    return "mono" if fit >= 85 else "partial" if fit >= 55 else "poly"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run1b_streamC4096")
    parser.add_argument("--banks_tag", default="prop1b")
    parser.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=220)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0,
                        help="label only the first N components (smoke test)")
    parser.add_argument("--env", type=Path,
                        default=Path(__file__).with_name(".env"))
    parser.add_argument("--evidence", default=None,
                        help="evidence filename (default evidence_<banks_tag>.json)")
    parser.add_argument("--catalog", default=None,
                        help="catalog filename (default catalog_<banks_tag>.json)")
    parser.add_argument("--only", default=None,
                        help="comma-separated component ids to label")
    args = parser.parse_args()
    run_dir = args.artifact_root / args.tag

    from dotenv import load_dotenv
    load_dotenv(args.env, override=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(f"no ANTHROPIC_API_KEY in {args.env}")
    import anthropic
    client = anthropic.Anthropic(max_retries=0)  # retries handled below

    evidence = json.loads(
        (run_dir / (args.evidence
                    or f"evidence_{args.banks_tag}.json")).read_text())
    cat_path = run_dir / (args.catalog or f"catalog_{args.banks_tag}.json")
    catalog = json.loads(cat_path.read_text()) if cat_path.exists() else {}
    lock = threading.Lock()
    usage = {"in": 0, "out": 0}

    def label_one(c):
        info = evidence[c]
        stats = {k: info[k] for k in ("fire_rate", "mean_share")}
        if not info["examples"]:
            return c, {"label": "dead/no-evidence", "category": "other",
                       "mono": "poly", "confidence": "high", **stats}
        def make_body(examples):
            return ("\n\n".join(f"[posterior {e['share']}] ...{e['text']}..."
                                for e in examples)
                    + f"\n\n({len(examples)} examples, each from a different "
                      f"source document.)")

        # A handful of components draw evidence windows that trip the API's
        # safety classifier (stop_reason "refusal", empty content). The signal
        # is in the pattern, not any one window, so fall back to disjoint
        # slices of the same evidence before giving up.
        n = args.examples
        pool_ex = info["examples"]
        bodies = [make_body(pool_ex[:n])]
        if len(pool_ex) > n:
            bodies.append(make_body(pool_ex[n:2 * n]))
        bodies.append(make_body(pool_ex[:n:2]))
        refusals = 0
        for attempt in range(args.max_retries):
            body = bodies[min(refusals, len(bodies) - 1)]
            try:
                resp = client.messages.create(
                    model=args.model, max_tokens=args.max_tokens,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": body}])
                with lock:
                    usage["in"] += resp.usage.input_tokens
                    usage["out"] += resp.usage.output_tokens
                if resp.stop_reason == "refusal" or not resp.content:
                    refusals += 1
                    if refusals >= len(bodies):
                        return c, {"label": "refused/unlabeled",
                                   "category": "other", "mono": "unknown",
                                   "confidence": "low", **stats}
                    continue
                # content may lead with a non-text block, or the JSON may be
                # truncated by max_tokens — take all text, then the last
                # balanced object in it.
                txt = "".join(b.text for b in resp.content
                              if getattr(b, "type", None) == "text").strip()
                start = txt.index("{")
                end = txt.rindex("}") + 1 if "}" in txt else -1
                if end <= start:  # truncated mid-object: close it
                    txt = txt[start:].rstrip().rstrip(",") + "}"
                    end = len(txt)
                    start = 0
                d = json.loads(txt[start:end])
                # The grade is defined as a function of `fit`, so recover it
                # whenever the model puts something else in the field (it
                # occasionally echoes the confidence value there instead).
                if d.get("mono") not in ("mono", "partial", "poly"):
                    d["mono"] = grade_from_fit(d.get("fit"))
                d.update(stats)
                return c, d
            except Exception as e:  # noqa: BLE001 — keep labeling the rest
                if attempt == args.max_retries - 1:
                    return c, {"label": f"ERROR {type(e).__name__}",
                               "category": "other", "mono": "poly",
                               "confidence": "low", **stats}
                time.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))

    keys = sorted(evidence, key=int)
    if args.only:
        want = set(args.only.split(","))
        keys = [c for c in keys if c in want]
    if args.limit:
        keys = keys[:args.limit]
    todo = [c for c in keys
            if not catalog.get(c) or catalog[c]["label"].startswith("ERROR")]
    log(f"labeling {len(todo)}/{len(keys)} components with {args.model} "
        f"({args.examples} examples each, {args.workers} workers)")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for c, d in pool.map(label_one, todo):
            catalog[c] = d
            done += 1
            if done % 64 == 0:
                cat_path.write_text(json.dumps(catalog, indent=1))
                rate = done / max(time.time() - t0, 1e-9)
                log(f"labeled {done}/{len(todo)} ({rate:.1f}/s, eta "
                    f"{(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min, "
                    f"{usage['in']:,} in / {usage['out']:,} out tokens)")
    cat_path.write_text(json.dumps(catalog, indent=1))

    graded = [d for d in catalog.values()
              if not d["label"].startswith("ERROR")
              and d["label"] != "refused/unlabeled"]
    counts = {g: sum(1 for d in graded if d.get("mono") == g)
              for g in ("mono", "partial", "poly")}
    n = max(len(graded), 1)
    log(f"catalog: {len(catalog)} components -> {cat_path}")
    log("monosemanticity: " + "  ".join(
        f"{g}={counts[g]} ({100 * counts[g] / n:.1f}%)" for g in counts))
    cats: dict[str, int] = {}
    for d in graded:
        cats[d.get("category", "other")] = cats.get(d.get("category"), 0) + 1
    log("categories: " + "  ".join(
        f"{k}={v}" for k, v in sorted(cats.items(), key=lambda x: -x[1])))
    n_err = sum(1 for d in catalog.values() if d["label"].startswith("ERROR"))
    log(f"errors: {n_err};  tokens: {usage['in']:,} in / {usage['out']:,} out")


if __name__ == "__main__":
    main()
