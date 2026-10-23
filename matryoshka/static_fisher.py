"""Static Fisher metric: per matrix, the target's output Fisher averaged over tokens, M = E_x E_{y~p(.|x)} [g g^T],
g = d(-log p(y))/d(matrix output) (fisher_attribution.py's cotangents, K sampled next tokens per position).

The Fisher sensor's score |r_c.h| sqrt(mean_k (g_k.u_c)^2) = |r_c.h| sqrt(u_c^T F_x u_c) needs K backwards per token;
with F_x replaced by its average M (fixed: the target is frozen), |r_c.h| sqrt(u_c^T M u_c) needs only the forward
that gives h. M is stored as L with M = L L^T (eigen-factor, so rank-deficient M is exact), d_out x d_out per matrix.
Estimated once from the last ROWS train rows (never reached by runs shorter than one epoch), fixed seed."""
import torch

from common import CONTEXT, MODULES

ROWS, FISHER_K, SEED = 128, 8, 4242


def estimate(model, data, rows=ROWS, k=FISHER_K, seed=SEED, microbatch=16):
    """[(L, eigenvalues)] in MODULES order: L [d_out, d_out] fp32 with L L^T = M, from data[-rows:]."""
    import numpy as np
    from fisher_attribution import capture_fisher
    block = torch.from_numpy(np.array(data[len(data)-rows:], dtype=np.int64)).cuda()
    sums, count = None, 0
    for first in range(0, rows, microbatch):
        cache = capture_fisher(model, block[first:first+microbatch], k=k, seed=seed + first, microbatch=microbatch)
        cots = [e['events']['cot'].double().flatten(0, 1) for e in cache]          # [n*k, d_out]
        sums = [c.T @ c for c in cots] if sums is None else [a + c.T @ c for a, c in zip(sums, cots)]
        count += len(cots[0])
        del cache, cots
    factors = []
    for m in sums:
        values, vectors = torch.linalg.eigh(m / count)
        values = values.clamp_min(0)
        factors.append(((vectors * values.sqrt()).float(), values.float()))
    return factors


def entries(cache, factors):
    """write_attribution.capture_inputs entries turned into metric_write entries (same h)."""
    return [dict(e, kind='metric_write', shared={'sensor': 'static_fisher', 'metric': L.to(e['h'].device)}) for e, (L, _) in zip(cache, factors)]


def cached(model, path):
    """The target's metric from `path`, estimated (estimate's defaults, as every run's) and saved there on first use:
    for scoring checkpoints that carry no static_metric (runs trained with another sensor)."""
    import os
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        from common import tokens, checkpoint
        data, _ = tokens()
        factors = estimate(model, data)
        checkpoint(dict(factors=[dict(L=L, eigenvalues=v) for L, v in factors], modules=list(MODULES)), path)
    return [(f['L'].cuda(), f['eigenvalues'].cuda()) for f in torch.load(path, map_location='cpu')['factors']]
