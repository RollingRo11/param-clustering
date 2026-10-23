"""Static Fisher metric: per matrix, the target's output Fisher averaged over tokens, M = E_x E_{y~p(.|x)} [g g^T],
g = d(-log p(y))/d(matrix output) (fisher_attribution.py's cotangents, K sampled next tokens per position).

The Fisher sensor's score |r_c.h| sqrt(mean_k (g_k.u_c)^2) = |r_c.h| sqrt(u_c^T F_x u_c) needs K backwards per token;
with F_x replaced by its average M (fixed: the target is frozen), |r_c.h| sqrt(u_c^T M u_c) needs only the forward
that gives h. M is stored as L with M = L L^T (eigen-factor, so rank-deficient M is exact), d_out x d_out per matrix.
Estimated once from the last ROWS train rows (never reached by runs shorter than one epoch), fixed seed."""
import torch

from common import CONTEXT, MODULES

ROWS, FISHER_K, SEED = 128, 8, 4242


def estimate(model, data, rows=ROWS, k=FISHER_K, seed=SEED, microbatch=16, gim=False):
    """[(L, eigenvalues)] in MODULES order: L [d_out, d_out] fp32 with L L^T = M, from data[-rows:]. gim: the cotangents
    of the GIM rules (gim_attribution.RULES: T=2 softmax, Q/K/V scaling, frozen RMSNorm), an LRP-style backward."""
    import numpy as np
    from fisher_attribution import capture_fisher
    block = torch.from_numpy(np.array(data[len(data)-rows:], dtype=np.int64)).cuda()
    sums, count = None, 0
    for first in range(0, rows, microbatch):
        cache = capture_fisher(model, block[first:first+microbatch], k=k, seed=seed + first, microbatch=microbatch, gim=gim)
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


def entries(cache, factors, kind='metric_write'):
    """write_attribution.capture_inputs entries turned into metric_write (or metric_interact) entries (same h)."""
    return [dict(e, kind=kind, shared={'sensor': 'static_fisher', 'metric': L.to(e['h'].device)}) for e, (L, _) in zip(cache, factors)]


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


_gram = {}


def greedy_scores(cache, pieces, steps, fixed=(), scales=None):
    """Greedy (matching-pursuit) ranking in the metric, for KLKeep: per token, start from nothing kept and add, `steps`
    times, the piece (over all matrices) whose addition most shrinks sum_m ||y_m - kept_m||^2, y_m = sum_c v_c the
    matrix's whole output in the metric, v_c = a_c L^T u_c; gain_c = 2 v_c.(y - kept) - ||v_c||^2, updated after each
    pick through the block-diagonal Gram G = U^T M U (no interactions across matrices). The remaining pieces follow in
    the order of their static score |v_c|. cache: metric_write entries. Returns per-matrix scores [n, C_m] whose
    descending order across matrices is that ranking (picks first, in pick order). float64.
    fixed: matrix indices whose pieces compete with their own static cost ||v_c||^2 (no interactions, never updated);
    scales: {matrix index: factor} multiplying that matrix's metric (its Gram block and its gains)."""
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        scales = scales or {}
        key = (tuple(p.read.data_ptr() for p in pieces), tuple(sorted(scales.items())))
        if key not in _gram:
            blocks = [(e['shared']['metric'].double().T @ p.write.double()) for e, p in zip(cache, pieces)]   # L^T U
            _gram.clear()
            _gram[key] = torch.block_diag(*[scales.get(j, 1.) * (b.T @ b) for j, b in enumerate(blocks)])     # [N, N]
        gram = _gram[key]
        a = torch.cat([e['h'].double() @ p.read.double() for e, p in zip(cache, pieces)], -1)            # [n, N]
        diag = gram.diagonal()
        static = a.abs() * diag.clamp_min(0).sqrt()
        gains = 2 * a * (a @ gram) - a.square() * diag
        score = static / static.max(-1, keepdim=True).values.clamp_min(1e-300)                           # in [0, 1]
        counts = [p.read.shape[1] for p in pieces]
        frozen = torch.zeros(sum(counts), dtype=torch.bool, device=a.device)
        for j in fixed: frozen[sum(counts[:j]):sum(counts[:j+1])] = True
        gains = torch.where(frozen, a.square() * diag, gains)            # fixed matrices: the cost of leaving c out
        rows = torch.arange(len(a), device=a.device)
        for step in range(steps):
            pick = gains.argmax(-1)
            score[rows, pick] = 1. + steps - step
            gains -= torch.where(frozen, 0., 2 * a * a[rows, pick, None] * gram[pick])
            gains[rows, pick] = float('-inf')
        return list(score.float().split(counts, -1))
