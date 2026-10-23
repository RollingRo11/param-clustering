"""Fit one score scale per matrix (12 numbers) for a frozen par-auto checkpoint: the global top-k ranks
exp(beta_m) * score instead of score, beta chosen by coordinate search to minimise the hard top-k KL at the given budgets on
a fit slice of the validation rows (default rows 2000-2031, disjoint from KLKeep's 0-127). Scores: the static sensor
(the checkpoint's own metric, or the target's cached one). Prints the fitted betas, as --matrix-log-scale for
evaluate_checkpoint.py. Usage: fit_matrix_scales.py CHECKPOINT [--budgets 71] [--rows 2000:2032]"""
import argparse
import json
import os

import numpy as np
import torch

import parauto_core as core
import static_fisher
from common import MODULES, target, tokens
from write_attribution import capture_inputs
import parauto_structured as structured


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('checkpoint'); ap.add_argument('--budgets', default='71')
    ap.add_argument('--rows', default='2000:2032'); ap.add_argument('--rounds', type=int, default=4)
    args = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = False
    budgets = [int(k) for k in args.budgets.split(',')]
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
            cache = static_fisher.entries(capture_inputs(model, rows), metric)
            log = torch.cat([structured.scores(e, p).float() for e, p in zip(cache, pieces)], -1).clamp_min(1e-30).log()
            logp = model(ids)[0].float().log_softmax(-1)
        batches.append((ids, log, logp))
    owner = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).cuda()

    def kl(beta):
        total, n = 0., 0
        with torch.no_grad():
            for ids, log, logp in batches:
                shifted = log + beta[owner]
                for k in budgets:
                    keep = torch.zeros_like(shifted).scatter_(-1, shifted.topk(k, -1).indices, 1.).split(counts, -1)
                    hooks = []
                    try:
                        for name, p, m in zip(MODULES, pieces, keep):
                            hooks.append(model.get_submodule(name).register_forward_hook(
                                lambda _l, i, _o, p=p, m=m: ((i[0] @ p.read) * m.view(*i[0].shape[:-1], -1)) @ p.write.T))
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            out = model(ids)[0].float().log_softmax(-1)
                    finally:
                        for h in hooks: h.remove()
                    total += float((logp.exp() * (logp - out)).sum(-1).mean()); n += 1
        return total / n

    beta = torch.zeros(len(counts), device='cuda')
    best = kl(beta); print(json.dumps(dict(start_kl=best)), flush=True)
    for step in [1., .5, .25, .125][:args.rounds]:
        for j in range(len(counts)):
            for delta in (-2 * step, -step, step, 2 * step):
                trial = beta.clone(); trial[j] += delta
                value = kl(trial)
                if value < best: best, beta = value, trial
        print(json.dumps(dict(step=step, kl=best, beta=[round(float(b), 3) for b in beta])), flush=True)
    spec = ','.join(f'{n}={float(b):.4f}' for n, b in zip(MODULES, beta))
    print(json.dumps(dict(fitted_kl=best, matrix_log_scale=spec)), flush=True)


if __name__ == '__main__':
    main()
