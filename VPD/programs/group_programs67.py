"""Look at and score hand-written gate programs for VPD's CI > 0.1 co-activation groups (coactive_vpd67.py).

Each group i has programs/vpd_group_{i:02d}.py in the step format (READS, step(carry, prefix, upstream) -> (carry, gate)).
The gate (0..1, one number per token for the whole group) multiplies the group's own output (its pieces' read x write);
the program sees only token strings and document ids. VPD's own on/off: a group is on at a token when at least half of
its pieces have CI > 0.1 (CI from dump_group_ci67.py).

  look    python group_programs67.py look --group 7       (val rows 0-127: what is around the tokens where VPD has it on)
  score   python group_programs67.py score [--groups 1 2] (val rows 128-255: agreement with VPD and KL)

KL(intact || model with only this group changed), three gates: the program, VPD's own per-piece CI > 0.1 mask, and 0
(group removed). Reported over all positions and over the positions where VPD or the program has the group on.
Needs VPD/out/{vpd_published.json, vpd_pieces_all.pt, group_ci.pt, val.uint16} and matryoshka/data/model_step_99999.pt.
"""
import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from model67 import Model, TARGET

D = HERE.parent/'out'
EOT = '<|endoftext|>'
LOOK, TEST = range(0, 128), range(128, 256)


def tokenizer():
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    return Tokenizer.from_file(hf_hub_download('EleutherAI/gpt-neox-20b', 'tokenizer.json'))


def data():
    val = np.memmap(D/'val.uint16', dtype=np.uint16).reshape(-1, 513)[:256, :512].astype(np.int64)
    tok = tokenizer(); vocab = {}
    strings = [[vocab.setdefault(int(t), tok.decode([int(t)], skip_special_tokens=False)) for t in row] for row in val]
    dump = torch.load(D/'group_ci.pt', weights_only=False)
    sizes = [len(g['pieces']) for g in dump['groups']]
    ci = dump['ci'].float().view(256, 512, -1).split(sizes, -1)
    vpd_on = [(c > 0.1).float().mean(-1) >= 0.5 for c in ci]                 # [256, 512] per group
    return val, strings, dump['groups'], ci, vpd_on


class Prefix:
    """What a program sees at position t: token strings and document ids up to and including t."""
    class View:
        def __init__(self, items, t): self.items, self.t = items, t
        def __len__(self): return self.t + 1
        def __getitem__(self, i):
            if isinstance(i, slice): return self.items[:self.t + 1][i]
            if i < 0: i += self.t + 1
            if not 0 <= i <= self.t: raise IndexError(i)
            return self.items[i]

    def __init__(self, strings, docs, t):
        self.token_strings, self.document_ids = Prefix.View(strings, t), Prefix.View(docs, t)


def documents(strings):
    """A new document starts after each <|endoftext|> (the end-of-text token closes the document before it)."""
    out, d = [], 0
    for s in strings:
        out.append(d)
        if s == EOT: d += 1
    return out


def gates(program, strings):
    docs = documents(strings); carry = None; out = []
    for t in range(len(strings)):
        carry, g = program.step(carry, Prefix(strings, docs, t), {})
        out.append(float(g))
    return out


def load_program(i):
    path = HERE/f'vpd_group_{i:02d}.py'
    spec = importlib.util.spec_from_file_location(path.stem, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


def look(a):
    val, strings, groups, ci, vpd_on = data(); i = a.group; g = groups[i - 1]; on = vpd_on[i - 1]
    print(f"group {i}: {g['matrix']} {len(g['pieces'])} pieces")
    cur, prev, pos = collections.Counter(), collections.Counter(), collections.Counter()
    for r in LOOK:
        for t in on[r].nonzero().flatten().tolist():
            cur[strings[r][t]] += 1; pos[min(t, 9)] += 1
            prev[strings[r][t - 1] if t else '<start>'] += 1
    total = int(on[list(LOOK)].sum())
    print('on at', total, 'of', 128 * 512, 'tokens; positions (9 = 9+):', dict(sorted(pos.items())))
    every = collections.Counter(s for r in LOOK for s in strings[r])
    print('current tokens (on / all occurrences):', [(s, n, every[s]) for s, n in cur.most_common(a.top)])
    print('previous tokens:', [(s, n) for s, n in prev.most_common(a.top)])
    for r in list(LOOK)[:a.examples]:
        hits = on[r].nonzero().flatten().tolist()[:3]
        for t in hits: print('  ', repr(''.join(strings[r][max(0, t - 8):t])), '->', repr(strings[r][t]), 'pos', t)


def score(a):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    val, strings, groups, ci, vpd_on = data()
    model = Model(); sd = torch.load(TARGET, weights_only=True, map_location='cpu'); sd.setdefault('lm_head.weight', sd['wte.weight'])
    model.load_state_dict(sd); model = model.to(device).eval().requires_grad_(False)
    pieces = torch.load(D/'vpd_pieces_all.pt', weights_only=False)['pieces']
    chosen = a.groups or [i for i in range(1, len(groups) + 1) if (HERE/f'vpd_group_{i:02d}.py').exists()]
    rows = list(TEST); prog_gate, results = {}, {}
    for i in chosen:
        p = load_program(i); prog_gate[i] = torch.tensor([gates(p, strings[r]) for r in rows])
    kl = {i: {k: [] for k in ('program', 'vpd', 'removed')} for i in chosen}
    for b in range(0, len(rows), 8):
        rb = rows[b:b + 8]; ids = torch.from_numpy(val[rb]).to(device)
        with torch.no_grad():
            lp = model(ids).log_softmax(-1)
            for i in chosen:
                g = groups[i - 1]; mod = model.get_submodule(g['matrix'])
                V = pieces[g['matrix']]['V'][:, g['pieces']].to(device); U = pieces[g['matrix']]['U'][g['pieces']].to(device)
                masks = dict(program=prog_gate[i][b:b + 8, :, None].expand(-1, -1, len(g['pieces'])),
                             vpd=(ci[i - 1][rb] > 0.1).float(), removed=torch.zeros(len(rb), 512, len(g['pieces'])))
                for name, mask in masks.items():
                    mask = mask.to(device)
                    h = mod.register_forward_hook(lambda _m, x, o: o - ((x[0] @ V) * (1 - mask)) @ U)
                    lq = model(ids).log_softmax(-1); h.remove()
                    kl[i][name].append((lp.exp() * (lp - lq)).sum(-1).cpu())
        print(json.dumps(dict(rows=b + 8)), flush=True)
    for i in chosen:
        on = vpd_on[i - 1][rows]; pg = prog_gate[i] > 0.5
        where = on | pg
        res = dict(matrix=groups[i - 1]['matrix'], pieces=len(groups[i - 1]['pieces']), vpd_on=int(on.sum()), program_on=int(pg.sum()),
                   both_on=int((on & pg).sum()))
        for name, v in kl[i].items():
            v = torch.cat(v); res[f'kl_{name}_all'] = float(v.mean()); res[f'kl_{name}_where_on'] = float(v[where].mean()) if where.any() else 0.
        results[i] = res; print(json.dumps(dict(group=i, **{k: (round(x, 4) if isinstance(x, float) else x) for k, x in res.items()})), flush=True)
    out = D/'group_programs_score.json'
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update({str(i): r for i, r in results.items()}); out.write_text(json.dumps(old, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    l = sub.add_parser('look'); l.add_argument('--group', type=int); l.add_argument('--top', type=int, default=12); l.add_argument('--examples', type=int, default=6)
    s = sub.add_parser('score'); s.add_argument('--groups', type=int, nargs='*')
    a = p.parse_args(); dict(look=look, score=score)[a.cmd](a)
