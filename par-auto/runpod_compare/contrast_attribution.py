"""Natural-experiment sensor: score pieces against a real output contrast, never an ablation.

Zero-ablation asks "what if this piece's write were deleted", a state the model never visits.
Here the counterfactual is another token the model actually saw. For each token x we find a
neighbour x' from the same batch whose input to a given matrix is close, and whose model output
is genuinely different. The read difference is exact (both h are cached), the output difference
is real (both z are cached), and the map from one to the other is fit on many such on-manifold
pairs instead of being a Jacobian taken along a direction the data never takes.

v1, with every choice stated:

CAPTURE. One forward pass of the intact target under bf16 autocast, no gradients. Forward hooks
on all 24 MODULES store each matrix's input h(x) [n, d_in]; a forward pre-hook on `ln_f` stores
the residual stream just before the final norm, z(x) [n, d_model]. We stop at the residual and
not at the logits: the vocabulary is 50,277, so a logit-space map would be a 50,277 x d_out
matrix per matrix. W_U and ln_f are frozen and shared by every token, so the final residual is
the right-sized proxy for "the model's output changed".

PAIRING (per matrix m, so the pair depends on which matrix's input we are matching).
  * Candidates: a random subset of `candidates` (default 4096) token positions drawn once per
    call and shared by all 24 matrices, so per-matrix differences come from h and not from a
    different candidate draw.
  * Excluded for token x: x itself and its immediate neighbours in the same sequence
    (|position difference| <= exclude_window, default 1). Adjacent tokens share almost the whole
    residual stream, so they are cheap near-duplicates, not experiments.
  * Contrast threshold: a single scalar, the median of ||z(x) - z(x')|| over a random subsample
    of (token, candidate) pairs. One global median, because z does not depend on the matrix.
  * Choice: among candidates that are not excluded AND whose contrast exceeds the threshold,
    take the one with the highest cosine similarity of h. If no candidate qualifies, fall back
    to the largest-contrast candidate among the 32 nearest by cosine. Fallbacks are counted.

RIDGE. Per matrix, one linear map from that matrix's own output difference to the final residual
difference, fit over all pairs in fp32:
      min_M sum_pairs || M (y(x)-y(x')) - (z(x)-z(x')) ||^2 + lambda ||M||^2,   y = W h
with W the frozen target weight (these modules have no bias) and
      lambda = 1e-3 * trace(A) / d_out,  A = dY^T dY the Gram matrix,
i.e. one thousandth of the mean eigenvalue of A, so the ridge is scale-free and matters only in
directions the data barely explores. Solved as (A + lambda I) X = dY^T dZ with M = X^T.
Diagnostics report both the in-sample residual fraction and a held-out one (every 5th pair is
held out from a second fit): with few pairs relative to d_out the in-sample number is optimistic,
and the held-out number is the one to trust.

SCORE. For piece c (rank one: read column r_c, write column u_c) at token x,
      delta_rho_c = r_c . (h(x) - h(x'))          the piece's read difference, exact
      s_c(x) = | < M (delta_rho_c u_c), z(x)-z(x') > | / || z(x)-z(x') ||
i.e. how much of the ACTUAL output contrast this piece's write difference carries, in units of
residual-stream norm. Since M (delta_rho_c u_c) = delta_rho_c * (M u_c), this is one [C, d_model]
precompute per matrix followed by an elementwise product. Returned per matrix as [n, C], the
same shape and units convention as the other sensors, so evaluate_checkpoint can reshape to
[B, T, C] and take a global top-k across the 24 matrices.

LIMITS OF v1. Rank-one pieces only (rank-k grouping is not implemented). The map M is fit on the
same batch it scores. The score is a signed projection made positive by |.|: a piece that pushes
against the contrast scores the same as one that pushes with it.
"""
import math

import torch

from common import CONTEXT, MODULES


@torch.no_grad()
def capture_contrast(model, rows, microbatch=2):
    """Cache each matrix's input and the pre-ln_f residual for every token. No gradients."""
    n = len(rows)*CONTEXT
    hs, z, filled = None, None, 0
    for first in range(0, len(rows), microbatch):
        block = rows[first:first+microbatch]
        pre, final, hooks = {}, {}, []
        def retain(name):
            def hook(_m, inputs, _out): pre[name] = inputs[0]
            return hook
        for name in MODULES:
            hooks.append(model.get_submodule(name).register_forward_hook(retain(name)))
        hooks.append(model.ln_f.register_forward_pre_hook(
            lambda _m, args: final.__setitem__('z', args[0])))
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(block[:, :-1])
        finally:
            for hook in hooks: hook.remove()
        if hs is None:
            hs = [torch.empty(n, pre[name].shape[-1], dtype=torch.bfloat16, device='cuda') for name in MODULES]
            z = torch.empty(n, final['z'].shape[-1], dtype=torch.bfloat16, device='cuda')
        lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
        for name, buffer in zip(MODULES, hs):
            buffer[lo:hi] = pre[name].flatten(0, 1).bfloat16()
        z[lo:hi] = final['z'].flatten(0, 1).bfloat16()
        filled += len(block)*CONTEXT
        del pre, final
    assert filled == n
    return hs, z


def _contrast_threshold(z, candidate, generator, sample=256):
    """Median ||z_i - z_j|| over a random subsample of (token, candidate) pairs."""
    rows = torch.randperm(len(z), generator=generator, device=z.device)[:min(sample, len(z))]
    d = torch.cdist(z[rows].float(), z[candidate].float())
    return float(d.flatten().median())


def choose_pairs(h, z, candidate, sequence, position, threshold, *, exclude_window=1,
                 chunk=512, fallback_k=32):
    """Nearest neighbour in h among candidates whose output contrast clears `threshold`.

    Returns partner indices [n] into the token axis plus per-token cosine, contrast and a
    bool flag marking tokens that had to fall back to the top-`fallback_k` rule.
    """
    n = len(h)
    hc = torch.nn.functional.normalize(h[candidate].float(), dim=-1)
    zc = z[candidate].float()
    cand_seq, cand_pos = sequence[candidate], position[candidate]
    partner = torch.empty(n, dtype=torch.long, device=h.device)
    cosine = torch.empty(n, dtype=torch.float32, device=h.device)
    contrast = torch.empty(n, dtype=torch.float32, device=h.device)
    fell_back = torch.zeros(n, dtype=torch.bool, device=h.device)
    for lo in range(0, n, chunk):
        hi = min(lo+chunk, n)
        cos = torch.nn.functional.normalize(h[lo:hi].float(), dim=-1) @ hc.T          # [c, S]
        dist = torch.cdist(z[lo:hi].float(), zc)                                      # [c, S]
        blocked = ((sequence[lo:hi, None] == cand_seq[None, :]) &
                   ((position[lo:hi, None]-cand_pos[None, :]).abs() <= exclude_window))
        cos = cos.masked_fill(blocked, -math.inf)
        allowed = (dist > threshold) & ~blocked
        pick = cos.masked_fill(~allowed, -math.inf).argmax(-1)
        empty = ~allowed.any(-1)
        if bool(empty.any()):
            rows = empty.nonzero(as_tuple=True)[0]
            near = cos[rows].topk(min(fallback_k, cos.shape[1]), dim=-1).indices       # [r, k]
            best = dist[rows].gather(-1, near).argmax(-1)
            pick[rows] = near.gather(-1, best[:, None]).squeeze(-1)
            fell_back[lo:hi] = empty
        index = torch.arange(hi-lo, device=h.device)
        partner[lo:hi] = candidate[pick]
        cosine[lo:hi] = cos[index, pick].clamp(min=-1., max=1.)
        contrast[lo:hi] = dist[index, pick]
        del cos, dist, blocked, allowed
    return partner, cosine, contrast, fell_back


def ridge_fit(dy, dz, ridge=1e-3, holdout=5):
    """Solve (A + lambda I) X = dY^T dZ in fp32 with lambda = ridge * trace(A)/d_out.

    Returns X [d_out, d_model] (so M = X^T) and a diagnostics dict. `holdout` keeps every
    holdout-th pair out of a second fit and reports the residual fraction on those pairs;
    holdout=0 skips it.
    """
    dy, dz = dy.float(), dz.float()

    def solve(a_rows, b_rows):
        A = a_rows.T @ a_rows
        lam = ridge*float(A.diagonal().sum())/A.shape[0]
        A.diagonal().add_(lam)
        X = torch.linalg.solve(A, a_rows.T @ b_rows)
        del A
        return X, lam

    X, lam = solve(dy, dz)
    total = float(dz.pow(2).sum())
    residual = float((dz - dy @ X).pow(2).sum())
    diagnostics = dict(pairs=int(len(dy)), d_out=int(dy.shape[1]), d_model=int(dz.shape[1]),
                       ridge_lambda=lam, residual_fraction=residual/max(total, 1e-30))
    if holdout and len(dy) >= 2*holdout:
        mask = torch.zeros(len(dy), dtype=torch.bool, device=dy.device)
        mask[::holdout] = True
        Xt, _ = solve(dy[~mask], dz[~mask])
        out_total = float(dz[mask].pow(2).sum())
        out_residual = float((dz[mask] - dy[mask] @ Xt).pow(2).sum())
        diagnostics['heldout_pairs'] = int(mask.sum())
        diagnostics['heldout_residual_fraction'] = out_residual/max(out_total, 1e-30)
        del Xt
    return X, diagnostics


def piece_scores(X, read, write, dh, dz, chunk=2048):
    """s_c = |(r_c . dh) * <X^T u_c, dz/||dz||>| for every token and piece; [n, C] fp32."""
    read, write, X = read.float(), write.float(), X.float()
    direction = torch.nn.functional.normalize(dz.float(), dim=-1)                     # [n, d_model]
    gain = write.T @ X                                                                # [C, d_model]
    out = torch.empty(len(dh), read.shape[1], dtype=torch.float32, device=read.device)
    for lo in range(0, len(dh), chunk):
        hi = min(lo+chunk, len(dh))
        rho = dh[lo:hi].float() @ read                                                # [c, C]
        out[lo:hi] = (rho*(direction[lo:hi] @ gain.T)).abs_()
        del rho
    del gain, direction
    return out


@torch.no_grad()
def contrast_scores(model, rows, pieces, *, microbatch=2, candidates=4096, exclude_window=1,
                    ridge=1e-3, seed=0, chunk=512, sample=256, fallback_k=32, holdout=5):
    """Score every rank-one piece by how much of the real output contrast its write carries.

    Returns (scores, diagnostics): scores is a list over MODULES of [n, C] fp32 tensors with
    n = len(rows)*CONTEXT, ready to reshape to [B, T, C] for a global top-k.
    """
    if any(getattr(p, 'read', None) is None for p in pieces):
        raise ValueError('contrast_scores expects FreePieces with rank-one read/write columns')
    hs, z = capture_contrast(model, rows, microbatch=microbatch)
    n = len(hs[0])
    device = z.device
    generator = torch.Generator(device=device).manual_seed(seed)
    candidate = torch.randperm(n, generator=generator, device=device)[:min(candidates, n)]
    sequence = torch.arange(n, device=device)//CONTEXT
    position = torch.arange(n, device=device) % CONTEXT
    threshold = _contrast_threshold(z, candidate, generator, sample=sample)
    scores, per_matrix = [], {}
    for name, h, p in zip(MODULES, hs, pieces):
        module = model.get_submodule(name)
        if module.bias is not None:
            raise ValueError(f'{name} has a bias; v1 assumes y = W h')
        partner, cosine, contrast, fell_back = choose_pairs(
            h, z, candidate, sequence, position, threshold,
            exclude_window=exclude_window, chunk=chunk, fallback_k=fallback_k)
        dh = (h.float() - h[partner].float())
        dz = (z.float() - z[partner].float())
        weight = module.weight.detach().float()
        dy = dh @ weight.T
        X, fit = ridge_fit(dy, dz, ridge=ridge, holdout=holdout)
        del dy, weight
        scores.append(piece_scores(X, p.read, p.write, dh, dz))
        per_matrix[name] = dict(mean_cosine=float(cosine.mean()),
                                mean_contrast=float(contrast.mean()),
                                fallback_fraction=float(fell_back.float().mean()),
                                components=int(p.read.shape[1]), **fit)
        del dh, dz, X, partner, cosine, contrast, fell_back
        torch.cuda.empty_cache()
    del hs, z
    diagnostics = dict(sensor='contrast', tokens=int(n), pairs=int(n), candidates=int(len(candidate)),
                       exclude_window=exclude_window, contrast_threshold=threshold, ridge=ridge,
                       seed=seed, fallback_k=fallback_k, holdout_every=holdout,
                       mean_cosine=float(sum(v['mean_cosine'] for v in per_matrix.values())/len(per_matrix)),
                       mean_contrast=float(sum(v['mean_contrast'] for v in per_matrix.values())/len(per_matrix)),
                       mean_residual_fraction=float(sum(v['residual_fraction'] for v in per_matrix.values())/len(per_matrix)),
                       per_matrix=per_matrix)
    return scores, diagnostics
