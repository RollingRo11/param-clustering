"""Assemble the auto-interp dashboard payload and inline it into the page.

Two evidence budgets were labeled from the same decomposition: a 5M-position
sweep and a 1B-position sweep. Same components, same rubric, same model — only
the number of positions the top-32 examples were drawn from differs. So the
script is parameterised over which catalog to render, and when a comparison
catalog is supplied it also builds the paired grade-shift panel, which is the
only way to tell "the decomposition is 60% monosemantic" from "we did not look
at enough text to see the pattern".

    python3.12 autointerp_dashboard.py                     # the 1B run
    python3.12 autointerp_dashboard.py --preset 5m         # the original
"""
import argparse
import json
import re
from pathlib import Path

RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
GEO = Path("/workspace/param-clustering/geo-attribution")
SCRATCH = Path(__file__).parent

PRESETS = {
    "1b": dict(catalog="catalog_prop1b_1B.json",
               evidence="evidence_prop1b_1B.json",
               compare="catalog_prop1b.json",
               positions=1_000_017_920, positions_label="1.00B",
               sweep="autointerp_sweep.py (dual-GPU, top-32/component)",
               out="out/autointerp_dashboard_1B.html"),
    "5m": dict(catalog="catalog_prop1b.json",
               evidence="evidence_prop1b.json",
               compare=None,
               positions=5_001_216, positions_label="5.00M",
               sweep="autointerp_stream.py (top-32/component)",
               out="out/autointerp_dashboard.html"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--preset", choices=sorted(PRESETS), default="1b")
ap.add_argument("--run", type=Path, default=RUN)
ap.add_argument("--out", type=Path)
args = ap.parse_args()
P = PRESETS[args.preset]

catalog = json.loads((args.run / P["catalog"]).read_text())
evidence = json.loads((args.run / P["evidence"]).read_text())
survey = [str(r["c"]) for f in ("survey_A.json", "survey_B.json")
          for r in json.loads((SCRATCH / f).read_text())]

GRADES = ("mono", "partial", "poly")
PRE, POST = 64, 34


def trim(text):
    """Keep a window around the «marked» token so rows stay one line-ish."""
    m = re.search("«(.*?)»", text, re.S)
    if not m:
        return text[:PRE + POST]
    pre, cur, post = text[:m.start()], m.group(1), text[m.end():]
    pre = ("…" + pre[-PRE:]) if len(pre) > PRE else pre
    post = (post[:POST] + "…") if len(post) > POST else post
    return f"{pre}«{cur}»{post}"


rows = []
for c, d in sorted(catalog.items(), key=lambda x: int(x[0])):
    ex = [{"p": e["share"], "t": trim(e["text"])}
          for e in evidence[c]["examples"][:4]]
    rows.append({
        "c": int(c), "l": d["label"], "cat": d["category"],
        "g": d.get("mono", "unknown"), "f": d.get("fit"),
        # a component whose grade never came back has no fit/confidence either
        "cf": d.get("confidence"), "fr": round(d["fire_rate"] * 100, 4),
        "ms": round(d["mean_share"] * 100, 4), "ex": ex,
    })

graded = [r for r in rows if r["g"] in GRADES]
overall = {g: sum(1 for r in graded if r["g"] == g) for g in GRADES}

sset = set(survey)
sub = [r for r in graded if str(r["c"]) in sset]
calibration = [
    {"who": "Human graders (2×, independent)", "n": 80,
     "counts": {"mono": 64, "partial": 14, "poly": 2}},
    {"who": "claude-sonnet-5 (this run)", "n": len(sub),
     "counts": {g: sum(1 for r in sub if r["g"] == g) for g in GRADES}},
]

cats = {}
for r in graded:
    cats.setdefault(r["cat"], {g: 0 for g in GRADES})[r["g"]] += 1
by_cat = sorted(({"cat": k, "counts": v, "n": sum(v.values())}
                 for k, v in cats.items()), key=lambda x: -x["n"])

fits = [r["f"] for r in graded if isinstance(r["f"], (int, float))]
hist = [0] * 20
for f in fits:
    hist[min(int(f) // 5, 19)] += 1

# ---- paired grade shift against the smaller evidence budget ----
shift = None
if P["compare"]:
    prev = json.loads((args.run / P["compare"]).read_text())
    pair = [c for c in catalog
            if c in prev and prev[c].get("mono") in GRADES
            and catalog[c].get("mono") in GRADES]
    matrix = {g: {h: 0 for h in GRADES} for g in GRADES}
    for c in pair:
        matrix[prev[c]["mono"]][catalog[c]["mono"]] += 1
    rank = {g: i for i, g in enumerate(GRADES)}
    better = sum(v for g, r in matrix.items() for h, v in r.items()
                 if rank[h] < rank[g])
    worse = sum(v for g, r in matrix.items() for h, v in r.items()
                if rank[h] > rank[g])
    pf = [prev[c]["fit"] for c in pair
          if isinstance(prev[c].get("fit"), (int, float))]
    shift = {
        "from": "5M positions", "to": "1B positions",
        "n": len(pair), "matrix": matrix,
        "better": better, "same": len(pair) - better - worse, "worse": worse,
        "fromCounts": {g: sum(matrix[g].values()) for g in GRADES},
        "toCounts": {g: sum(matrix[x][g] for x in GRADES) for g in GRADES},
        "fitFrom": round(sum(pf) / len(pf), 1),
        "fitTo": round(sum(fits) / len(fits), 1),
    }

payload = {
    "rows": rows,
    "overall": overall,
    "graded": len(graded),
    "total": len(rows),
    "calibration": calibration,
    "byCat": by_cat,
    "hist": hist,
    "fitMean": round(sum(fits) / len(fits), 1),
    "german": [3634, 1668, 42, 2207, 1406, 2520, 928, 3600, 493, 2161, 2318],
    "emoticon": [3201, 1912, 1620, 3931],
    "induction": [3392, 108, 2747],
    "positions": P["positions"],
    "positionsLabel": P["positions_label"],
    "sweepScript": P["sweep"],
    "shift": shift,
    "meanExamples": round(
        sum(len(evidence[c]["examples"]) for c in evidence) / len(evidence), 1),
}

template = (SCRATCH / "dashboard_template.html").read_text()
out = args.out or (GEO / P["out"])
# Raw Pile text carries `</script>` and U+2028/9, which are legal JSON but
# terminate or break the inline script block.
blob = json.dumps(payload, separators=(",", ":"))
for bad, good in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"),
                  (" ", "\\u2028"), (" ", "\\u2029")):
    blob = blob.replace(bad, good)
out.write_text(template.replace("/*__PAYLOAD__*/null", blob))
print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
print("overall:", overall, "graded", len(graded), "of", len(rows))
print("calibration:", json.dumps(calibration))
print("categories:", [(c["cat"], c["n"]) for c in by_cat])
if shift:
    print(f"shift: {shift['better']} sharpened, {shift['same']} unchanged, "
          f"{shift['worse']} degraded  (fit {shift['fitFrom']} -> "
          f"{shift['fitTo']})")
