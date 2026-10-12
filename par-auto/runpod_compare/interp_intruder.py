"""Label-free intruder test (as geo-attribution/intruder_detection.py) on interp_render.py output.

4 windows where one component fires + 1 top window of a different component; a judge picks
the odd one out. Chance is 20%. Same prompt and trial rules as the geo/VPD runs.
Needs ANTHROPIC_API_KEY in the environment.

  uv run --no-project --with anthropic python interp_intruder.py <evidence_text.json> [--n 400]
"""
import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = """You are given 5 short text excerpts. In each, one token is marked with «guillemets».

Four of them were selected because the SAME hidden feature of a language model fires on the marked token. Exactly one is an INTRUDER: its marked token comes from a different feature entirely.

Look at what is common to the marked tokens — the token itself, the surrounding syntax, the position in a construction, the document genre. Then decide which single excerpt does not share that property.

Reply with ONLY a JSON object:
{"intruder": <1-5>, "reason": "<8 words or fewer>", "confidence": "high"|"medium"|"low"}"""


def trim(text, pre=90, post=45):
    m = re.search("«(.*?)»", text, re.S)
    if not m: return text[:pre+post]
    a, cur, b = text[:m.start()], m.group(1), text[m.end():]
    return f"{('…'+a[-pre:]) if len(a) > pre else a}«{cur}»{(b[:post]+'…') if len(b) > post else b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('evidence'); ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--model', default='claude-sonnet-5'); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic(max_retries=0)
    ev = json.loads(Path(args.evidence).read_text())['components']
    usable = [c for c in ev if len(ev[c]['examples']) >= 6]
    rng = random.Random(args.seed); rng.shuffle(usable)
    picks, lock, usage = usable[:args.n], threading.Lock(), dict(inp=0, out=0, err=0)

    def trial(c):
        r = random.Random(f'{args.seed}:{c}')
        real = r.sample(ev[c]['examples'][:12], 4)
        other = c
        while other == c: other = r.choice(usable)
        items = [(t['text'], False) for t in real] + [(r.choice(ev[other]['examples'][:8])['text'], True)]
        r.shuffle(items)
        truth = 1 + next(i for i, (_, x) in enumerate(items) if x)
        body = '\n\n'.join(f'{i+1}. ...{trim(t)}...' for i, (t, _) in enumerate(items))
        for _ in range(4):
            try:
                resp = client.messages.create(model=args.model, max_tokens=120, system=SYSTEM,
                                              messages=[{'role': 'user', 'content': body}])
                with lock: usage['inp'] += resp.usage.input_tokens; usage['out'] += resp.usage.output_tokens
                txt = ''.join(b.text for b in resp.content if getattr(b, 'type', None) == 'text')
                guess = int(json.loads(txt[txt.index('{'):txt.rindex('}')+1])['intruder'])
                if 1 <= guess <= 5:
                    return dict(component=int(c), matrix=ev[c]['matrix'], truth=truth, guess=guess,
                                correct=guess == truth, intruder_from=int(other))
            except Exception:
                continue
        with lock: usage['err'] += 1

    with ThreadPoolExecutor(args.workers) as pool:
        rows = [r for r in pool.map(trial, picks) if r]
    by_type = {}
    for r in rows: by_type.setdefault(r['matrix'].split('.')[-1], []).append(r['correct'])
    out = dict(format='intruder_detection_v1', model=args.model, n_trials=len(rows), chance=.2,
               accuracy=round(sum(r['correct'] for r in rows)/max(1, len(rows)), 4),
               by_matrix_type={k: dict(n=len(v), accuracy=round(sum(v)/len(v), 4)) for k, v in by_type.items()},
               errors=usage['err'], tokens=dict(input=usage['inp'], output=usage['out']), trials=rows)
    target = Path(args.evidence).with_name(Path(args.evidence).stem.replace('_text', '')+'_intruder.json')
    target.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'trials'}, indent=1))


if __name__ == '__main__': main()
