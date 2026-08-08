"""VPD Section 3.4, Figure 6: intruder detection.

The 81.7% monosemanticity figure comes from the labeller grading its own
labels, which is exactly the kind of number that should not be trusted on its
own. Intruder detection is the field-standard alternative and it is LABEL-FREE:

    show a judge 5 text windows — 4 on which one component fires, and 1
    "intruder" drawn from a different component — and ask which does not
    belong. Chance is 20%. Accuracy measures whether the component's
    activation set is coherent, with no reference to any written label.

Two things this buys beyond replicating VPD:

  independence   a component can score well here even if its label is bad, and
                 badly even if its label reads well, so it is a genuine second
                 opinion on the decomposition rather than on the labelling.

  validation     stratifying by the grade the labeller assigned turns the
                 self-graded mono/partial/poly scale into a testable claim: if
                 the grades mean anything, intruder accuracy must fall across
                 them.

The intruder is drawn from another component's TOP examples, not from random
text, so it is a fluent, high-activation window for something else — the hard
version of the task.

    python3.12 intruder_detection.py --n 400 --workers 24
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geo1b  # noqa: F401
from german_vpd_1b import log

SYSTEM = """You are given 5 short text excerpts. In each, one token is marked with «guillemets».

Four of them were selected because the SAME hidden feature of a language model fires on the marked token. Exactly one is an INTRUDER: its marked token comes from a different feature entirely.

Look at what is common to the marked tokens — the token itself, the surrounding syntax, the position in a construction, the document genre. Then decide which single excerpt does not share that property.

Reply with ONLY a JSON object:
{"intruder": <1-5>, "reason": "<8 words or fewer>", "confidence": "high"|"medium"|"low"}"""


def trim(text, pre=90, post=45):
    m = re.search("«(.*?)»", text, re.S)
    if not m:
        return text[:pre + post]
    a, cur, b = text[:m.start()], m.group(1), text[m.end():]
    a = ("…" + a[-pre:]) if len(a) > pre else a
    b = (b[:post] + "…") if len(b) > post else b
    return f"{a}«{cur}»{b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1b_streamC4096")
    ap.add_argument("--artifact_root", type=Path, default=geo1b.SHM_ROOT)
    ap.add_argument("--evidence", default="evidence_prop1b_1B.json")
    ap.add_argument("--catalog", default="catalog_prop1b_1B.json")
    ap.add_argument("--env", type=Path,
                    default=Path("/workspace/param-clustering/geo-attribution/.env"))
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k_real", type=int, default=4)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max_tokens", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--difficulty", choices=("random", "near"),
                    default="random",
                    help="where the intruder comes from: a random component, "
                         "or the target's nearest neighbour in centroid space")
    ap.add_argument("--neighbour_rank", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="intruder_detection.json")
    args = ap.parse_args()
    run_dir = args.artifact_root / args.tag

    from dotenv import load_dotenv
    load_dotenv(args.env, override=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(f"no ANTHROPIC_API_KEY in {args.env}")
    import anthropic
    client = anthropic.Anthropic(max_retries=0)

    evidence = json.loads((run_dir / args.evidence).read_text())
    catalog = json.loads((run_dir / args.catalog).read_text())
    usable = [c for c in evidence
              if len(evidence[c]["examples"]) >= args.k_real + 2
              and catalog.get(c, {}).get("mono") in ("mono", "partial", "poly")]
    rng = random.Random(args.seed)
    rng.shuffle(usable)
    picks = usable[:args.n]
    log(f"{len(picks)} components with enough evidence (of {len(usable)})")

    # nearest-neighbour components in the frozen centroid space. If a
    # component cannot be told apart from its nearest neighbour, the two are
    # splits of one feature — VPD's feature-splitting question, asked with a
    # judge instead of an alive-count.
    neighbour = None
    if args.difficulty == "near":
        import torch
        from streaming_decomposition import load_stream_model
        sm = load_stream_model(run_dir / "stream_model.pt", "cpu")
        Ctr = torch.nn.functional.normalize(sm["centroids"].float(), dim=1)
        sim = Ctr @ Ctr.t()
        sim.fill_diagonal_(-2)
        nn = sim.topk(args.neighbour_rank, dim=1).indices[:, -1]
        neighbour = {str(i): str(int(j)) for i, j in enumerate(nn)}
        log(f"nearest-neighbour intruders (rank {args.neighbour_rank}); "
            f"mean cosine {float(sim.max(1).values.mean()):.4f}")

    lock = threading.Lock()
    usage = {"in": 0, "out": 0, "err": 0}

    def trial(c):
        r = random.Random(f"{args.seed}:{c}")
        real = r.sample(evidence[c]["examples"][:12], args.k_real)
        if neighbour is not None:
            other = neighbour[c]
            if other not in evidence or not evidence[other]["examples"]:
                return None
        else:
            other = c
            while other == c:
                other = r.choice(usable)
        intr = r.choice(evidence[other]["examples"][:8])
        items = [(t["text"], False) for t in real] + [(intr["text"], True)]
        r.shuffle(items)
        truth = 1 + next(i for i, (_, is_i) in enumerate(items) if is_i)
        body = "\n\n".join(f"{i + 1}. ...{trim(t)}..."
                           for i, (t, _) in enumerate(items))
        for _ in range(args.max_retries):
            try:
                resp = client.messages.create(
                    model=args.model, max_tokens=args.max_tokens,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": body}])
                with lock:
                    usage["in"] += resp.usage.input_tokens
                    usage["out"] += resp.usage.output_tokens
                if resp.stop_reason == "refusal" or not resp.content:
                    continue
                txt = "".join(b.text for b in resp.content
                              if getattr(b, "type", None) == "text").strip()
                d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
                guess = int(d["intruder"])
                if not 1 <= guess <= 5:
                    continue
                return {"component": int(c), "truth": truth, "guess": guess,
                        "correct": guess == truth,
                        "confidence": d.get("confidence"),
                        "reason": d.get("reason"),
                        "grade": catalog[c]["mono"],
                        "fit": catalog[c].get("fit"),
                        "category": catalog[c]["category"],
                        "intruder_from": int(other)}
            except Exception:
                continue
        with lock:
            usage["err"] += 1
        return None

    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(trial, picks), 1):
            if r:
                rows.append(r)
            if i % 50 == 0:
                acc = sum(x["correct"] for x in rows) / max(len(rows), 1)
                log(f"{i}/{len(picks)}  running accuracy {acc:.3f} "
                    f"({usage['in']:,} in / {usage['out']:,} out)")

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / max(n, 1)
    by_grade = {}
    for g in ("mono", "partial", "poly"):
        sub = [r for r in rows if r["grade"] == g]
        if sub:
            by_grade[g] = {"n": len(sub),
                           "accuracy": round(sum(x["correct"] for x in sub)
                                             / len(sub), 4)}
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["correct"])
    by_cat = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
              for k, v in sorted(by_cat.items(), key=lambda kv: -len(kv[1]))}
    # guess distribution: a judge that always says "3" would look good if the
    # truth were not uniform, so check both
    gd = {i: sum(1 for r in rows if r["guess"] == i) for i in range(1, 6)}
    td = {i: sum(1 for r in rows if r["truth"] == i) for i in range(1, 6)}

    out = {"format": "intruder_detection_v1", "model": args.model,
           "difficulty": args.difficulty,
           "neighbour_rank": args.neighbour_rank,
           "n_trials": n, "k_real": args.k_real, "chance": 0.2,
           "accuracy": round(acc, 4),
           "by_grade": by_grade, "by_category": by_cat,
           "guess_distribution": gd, "truth_distribution": td,
           "errors": usage["err"], "tokens": {"in": usage["in"],
                                              "out": usage["out"]},
           "trials": rows}
    (run_dir / args.out).write_text(json.dumps(out, indent=1))
    log(f"\nintruder detection accuracy {acc:.4f} on {n} trials "
        f"(chance 0.20)")
    for g, v in by_grade.items():
        log(f"  {g:<8} {v['accuracy']:.4f}  (n={v['n']})")
    log(f"guesses {gd}  truths {td}")
    log(f"wrote {run_dir / args.out}")


if __name__ == "__main__":
    main()
