"""Which pieces of published VPD switch on together, within one matrix (the VPD twin of coactive67.py).

A VPD piece is "on" at a token when its causal importance (lower_leaky, as ci_census.py) is above a threshold.
Pass 1 counts, per matrix, the tokens each piece is on and the tokens each pair is on together; pairs whose token sets
nearly coincide (Jaccard >= 0.9, both on for >= 50 tokens) are joined into groups. Pass 2 records where each group is
all on (position in the context, current and previous token id), so a group can be read off.
Usage (from code/vpd, like slurm_ci_census.sh): coactive_vpd67.py --checkpoint vpd.pth --rows 256 --output OUT.json
--device cpu runs VPD's code on CPU (Module.cuda made a no-op) in fp32; on cuda it runs under bf16 autocast as ci_census.py.
"""
import argparse
import collections
import contextlib
import time
from pathlib import Path

import numpy as np
import torch

from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json

THRESHOLDS = (0., 0.1, 0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--rows', type=int, default=256)
    ap.add_argument('--min-support', type=int, default=50); ap.add_argument('--jaccard', type=float, default=0.9)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    dev = args.device
    if dev == 'cpu': torch.nn.Module.cuda = lambda self, *a, **k: self      # train_pair.VPD calls .cuda()
    precision = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dev == 'cuda' else contextlib.nullcontext
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    from train_pair import VPD
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'vpd' and saved['target_sha256'] == TARGET_SHA
    trainer = VPD(target(dev), evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
    trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
    completed = saved.get('completed'); del saved
    data, _ = tokens('val')
    start = time.time()

    def ci_batches():
        for first in range(0, args.rows, 2):
            rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device=dev); ids = rows[:, :-1]
            with torch.no_grad(), precision():
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
            yield first, ids, {n: ci.lower_leaky[n].float().flatten(0, 1) for n in MODULES}

    # pass 1: per threshold and matrix, on-counts and pair co-counts
    n = {t: {} for t in THRESHOLDS}; both = {t: {} for t in THRESHOLDS}; per_token = {t: 0. for t in THRESHOLDS}; N = 0
    for first, ids, ci in ci_batches():
        N += ids.numel()
        for t in THRESHOLDS:
            for m in MODULES:
                B = (ci[m] > t).float()                   # counts stay exact in fp32 (< 2^24)
                n[t][m] = n[t].get(m, 0) + B.sum(0); both[t][m] = both[t].get(m, 0) + B.T @ B
                per_token[t] += float(B.sum())
        if first % 32 == 0: print(dict(rows=first + 2, seconds=round(time.time() - start)), flush=True)

    result = dict(device=dev, checkpoint=str(args.checkpoint), completed=completed, tokens=N, rows=args.rows, min_support=args.min_support,
                  jaccard=args.jaccard, thresholds={})
    groups = []                                                     # (threshold, matrix, piece ids)
    for t in THRESHOLDS:
        out = dict(on_per_token=per_token[t] / N, matrices={}); best_all = []
        for m in MODULES:
            c, b = n[t][m], both[t][m]
            jac = b / (c[:, None] + c[None] - b).clamp_min(1); jac.fill_diagonal_(0)
            live = c >= args.min_support
            best_all.append(jac[live][:, live].max(1).values if live.sum() > 1 else jac.new_zeros(0))
            upper = torch.triu(torch.ones_like(jac, dtype=torch.bool), 1) & live[:, None] & live[None]
            nested = upper & (b / torch.minimum(c[:, None], c[None]).clamp_min(1) >= args.jaccard)
            hit = upper & (jac >= args.jaccard)
            pa, pb = (x.tolist() for x in hit.nonzero(as_tuple=True))
            parent = {}
            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for x, y in zip(pa, pb): parent[find(x)] = find(y)
            g = collections.defaultdict(list)
            for x in parent: g[find(x)].append(x)
            mg = sorted((sorted(v) for v in g.values()), key=len, reverse=True)
            groups += [(t, m, v) for v in mg]
            out['matrices'][m] = dict(pieces=int(c.numel()), live=int(live.sum()), always_on=int((c >= 0.5 * N).sum()),
                                      jaccard_pairs=len(pa), nested_pairs=int(nested.sum()),
                                      groups=[dict(pieces=v, support=[int(c[x]) for x in v]) for v in mg])
        best = torch.cat(best_all)
        out['live'] = int(best.numel())
        out['best_partner_jaccard_quantiles'] = {str(q): float(best.quantile(q)) for q in (.5, .9, .99, .999)} if best.numel() else {}
        out['pieces_with_partner_at_least'] = {str(x): int((best >= x).sum()) for x in (.5, .7, .8, .9)}
        result['thresholds'][str(t)] = out
        print(dict(threshold=t, on_per_token=out['on_per_token'], live=out['live'], q=out['best_partner_jaccard_quantiles'],
                   partners=out['pieces_with_partner_at_least']), flush=True)

    # pass 2: where each group is all on
    where = [dict(positions=collections.Counter(), current=collections.Counter(), previous=collections.Counter(), all_on=0)
             for _ in groups]
    if groups:
        for first, ids, ci in ci_batches():
            flat = ids.flatten(); pos = torch.arange(ids.shape[1], device=dev).repeat(ids.shape[0])
            for (t, m, v), w in zip(groups, where):
                hit = (ci[m][:, v] > t).all(1).nonzero().flatten()
                w['all_on'] += int(hit.numel())
                for i in hit.tolist():
                    p = int(pos[i]); w['positions'][min(p, 9)] += 1
                    w['current'][int(flat[i])] += 1
                    if p: w['previous'][int(flat[i - 1])] += 1
        for (t, m, v), w in zip(groups, where):
            for gr in result['thresholds'][str(t)]['matrices'][m]['groups']:
                if gr['pieces'] == v:
                    gr.update(all_on=w['all_on'], positions_capped_at_9=dict(w['positions'].most_common()),
                              current_token_ids=w['current'].most_common(10), previous_token_ids=w['previous'].most_common(10))
    result['seconds'] = time.time() - start
    save_json(result, args.output); print('COACTIVE_VPD_DONE', flush=True)


if __name__ == '__main__':
    main()
