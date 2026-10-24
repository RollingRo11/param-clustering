"""KL(intact || gated) and active pieces per token on SimpleStories val rows when each piece is gated by its own static
score against its learned floor (MATRYOSHKA_FLOORS runs): 'hard' = 1 above the floor, else 0; 'ramp w' = the leaky
hard-clipped ramp 0.5 + (score - floor) / (2 w floor) clipped to [0, 1] (MATRYOSHKA_FLOOR_RAMP's forward), active where > 0.
Usage: gate_kl_ss.py MODEL.pth [--ramps 0.05,0.1,0.2] [--rows 128] [--output OUT.json]"""
import argparse
import json
import os
import sys

os.environ.setdefault('MATRYOSHKA_TARGET', 'simplestories')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import parauto_core as core
import parauto_structured as structured
import static_fisher
from common import MODULES, target, tokens
from write_attribution import capture_inputs


def gate(score, floor, ramp):
    return (0.5 + (score - floor) / (2 * ramp * floor)).clamp(0, 1) if ramp else (score > floor).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint'); ap.add_argument('--ramps', default='0.05,0.1,0.2'); ap.add_argument('--rows', type=int, default=128)
    ap.add_argument('--output')
    args = ap.parse_args()
    m = target(); val, _ = tokens('val')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    metric = [(f['L'].cuda(), f['eigenvalues']) for f in saved['static_metric']]
    counts = [p.read.shape[1] for p in pieces]
    floors = saved['score_floor'].cuda().split(counts)
    gates = {'hard': 0.} | {f'ramp{w}': float(w) for w in args.ramps.split(',')}
    kl = {g: 0. for g in gates}; active = {g: torch.zeros(len(MODULES)) for g in gates}; batches = 0
    for first in range(0, args.rows, 8):
        rows = torch.from_numpy(np.array(val[first:first + 8], dtype=np.int64)).cuda(); ids = rows[:, :-1]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = [structured.scores(e, p).float() for e, p in zip(static_fisher.entries(capture_inputs(m, rows), metric), pieces)]
            logp = m(ids)[0].float().log_softmax(-1)
            for g, w in gates.items():
                masks = [gate(x, f, w) for x, f in zip(raw, floors)]
                hooks = [m.get_submodule(n).register_forward_hook(
                    lambda _l, i, _o, p=p, k=k: ((i[0] @ p.read) * k.view(*i[0].shape[:-1], -1)) @ p.write.T)
                    for n, p, k in zip(MODULES, pieces, masks)]
                try: out = m(ids)[0].float().log_softmax(-1)
                finally:
                    for h in hooks: h.remove()
                kl[g] += float((logp.exp() * (logp - out)).sum(-1).mean())
                active[g] += torch.stack([(k > 0).float().sum(-1).mean().cpu() for k in masks])
        batches += 1
    result = {g: dict(kl=kl[g] / batches, active=float(active[g].sum()) / batches,
                      active_per_matrix={n: float(a) / batches for n, a in zip(MODULES, active[g])}) for g in gates}
    for g, r in result.items(): print(json.dumps(dict(gate=g, kl=round(r['kl'], 4), active=round(r['active'], 1))), flush=True)
    if args.output:
        with open(args.output, 'w') as f: json.dump(dict(checkpoint=args.checkpoint, rows=args.rows, gates=result), f, indent=1)


if __name__ == '__main__': main()
