"""Per-piece score floors for a frozen par-auto checkpoint (static sensor), for KLKeep's --score-floor: rank
max(0, score_c - floor_c), so a piece only counts when it is above its own usual level (interference, the always-weakly-on
part of |r.h|, is subtracted). Floors are in static-score units.
  median: floor_c = the median over positions of piece c's score on the fit rows.
  learned (--fit): floor_c = median_c * exp(theta_c), theta fitted by Adam to the hard top-k KL at budgets 21/71/213
          on the fit rows, the ranking smoothed as softplus((score - floor)/tau)*tau with the sigmoid top-k Jacobian
          passed straight through (as training); the pieces stay frozen.
Fit rows default to val 2000-2063, disjoint from KLKeep's 0-127. Usage: floor_ranking.py CHECKPOINT OUT.pt [--fit]"""
import argparse
import json
import os

import numpy as np
import torch

import matryoshka67 as mt
import parauto_core as core
import parauto_structured as structured
import static_fisher
from common import MODULES, target, tokens
from write_attribution import capture_inputs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('checkpoint'); ap.add_argument('out')
    ap.add_argument('--rows', default='2000:2064'); ap.add_argument('--fit', action='store_true')
    ap.add_argument('--steps', type=int, default=300); ap.add_argument('--lr', type=float, default=0.05)
    args = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = False
    lo, hi = (int(x) for x in args.rows.split(':'))
    model = target(); val, _ = tokens('val')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    metric = ([(f['L'].cuda(), f['eigenvalues']) for f in saved['static_metric']] if 'static_metric' in saved else
              static_fisher.cached(model, os.environ.get('MATRYOSHKA_STATIC_CACHE', '/root/data/ss/static_fisher.pt')))
    counts = [p.read.shape[1] for p in pieces]
    batches = []
    for first in range(lo, hi, 8):
        rows = torch.from_numpy(np.array(val[first:min(first+8, hi)], dtype=np.int64)).cuda(); ids = rows[:, :-1]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            score = torch.cat([structured.scores(e, p).float() for e, p in zip(static_fisher.entries(capture_inputs(model, rows), metric), pieces)], -1)
            logp = model(ids)[0].float().log_softmax(-1)
        batches.append((ids, score, logp))
    median = torch.cat([b[1] for b in batches]).median(0).values.clamp_min(1e-12)
    result = dict(floor=median.cpu(), kind='median', rows=args.rows)
    print(json.dumps(dict(median_floor_mean=float(median.mean()))), flush=True)
    if args.fit:
        theta = torch.zeros_like(median, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=args.lr)
        tau = 0.1 * median
        budgets = (21, 71, 213)
        for step in range(args.steps):
            ids, score, logp = batches[step % len(batches)]
            k = budgets[step % len(budgets)]
            floor = median * theta.exp()
            above = torch.nn.functional.softplus((score - floor) / tau) * tau + 1e-6 * score
            mask, _ = mt.forward_mask(above.clamp_min(1e-30).log(), [k], 'hard')
            alphas = [a.reshape(*ids.shape, -1) for a in mask.split(counts, -1)]
            kl = mt.kl_per_position(logp, mt.masked_logits(model, ids, pieces, alphas)).mean()
            opt.zero_grad(); kl.backward(); opt.step()
            if step % 50 == 0 or step == args.steps - 1:
                print(json.dumps(dict(step=step, k=k, kl=float(kl), theta_mean=float(theta.mean()), theta_std=float(theta.std()))), flush=True)
        result = dict(floor=(median * theta.exp()).detach().cpu(), kind='learned', rows=args.rows, steps=args.steps, lr=args.lr)
    torch.save(result, args.out)
    per = [float(x.mean()) for x in (result['floor'].cuda() / median).log().split(counts)]
    print(json.dumps(dict(saved=args.out, log_floor_over_median_by_matrix=dict(zip(MODULES, [round(v, 3) for v in per])))), flush=True)


if __name__ == '__main__':
    main()
