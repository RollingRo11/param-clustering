"""Decode interp_evidence.py output into autointerp-style text (the firing token in «»)."""
import json
import sys
from pathlib import Path
from tokenizers import Tokenizer


def render(path):
    tok = Tokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    raw = json.loads(Path(path).read_text())
    out = {}
    for c, e in raw['components'].items():
        examples = []
        for x in e['examples']:
            ids, m = x['ids'], x['mark']
            examples.append(dict(score=x['score'], row=x['row'],
                                 text=f"{tok.decode(ids[:m])}«{tok.decode([ids[m]])}»{tok.decode(ids[m+1:])}"))
        item = dict(matrix=e['matrix'], index=e['index'], fire_rate=e['fire_rate'], examples=examples)
        for key in ('promotes', 'suppresses'):
            if key in e: item[key] = [tok.decode([i]) for i in e[key]]
        out[c] = item
    target = Path(path).with_name(Path(path).stem+'_text.json')
    target.write_text(json.dumps(dict({k: v for k, v in raw.items() if k != 'components'}, components=out), indent=1))
    return target


if __name__ == '__main__':
    for p in sys.argv[1:]: print(render(p))
