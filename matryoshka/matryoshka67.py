"""Matryoshka training of exact-sum rank-one pieces (after MAttr, arXiv 2609.25518), 5,000 steps.

par-auto's own read x write score (Fisher or GIM) picks which pieces each token keeps; a sigmoid top-k mask at a
random budget k keeps them; the loss is the KL of the actually masked model to the intact one. There is no gate
network; by default no Lp term and no PGD (MATRYOSHKA_LP_COEFF adds the old Lp term, MATRYOSHKA_ADV_COEFF adds VPD's
persistent PGD loss, see Adversary, MATRYOSHKA_MASS_COEFF a pull of each matrix's piece mass toward 100%, see mass_penalty). Everything else (init, data order, score cache, lr schedule,
exact-sum projection) matches the fisher_k4_5000 control (MATRYOSHKA_SENSOR=gim: the gim_1h control).

One update, 64 contexts x 512:
  1. intact score cache: Fisher = K=4 next tokens sampled from the model, one backward each (as the control);
     GIM = one GIM-rule backward of the true next-token CE (as gim_1h).
  2. per token: a_c = |r_c.h| * RMS_k(g_k.u_c) (GIM: |r_c.h| * |g.u_c|) over all 43,008 pieces,
     s = log a, alpha = sigmoid(s + c) with c solved per token so sum_c alpha = k (sigmoid top-k,
     Wijk et al. 2025). k drawn with density proportional to k^-BUDGET_POWER on [K_MIN, N-1] (power 1 =
     log-uniform, the earlier runs), one draw per BUDGET_CONTEXTS contexts (4 in the earlier runs), optionally
     stratified: one draw from each equal-probability slice per update, shuffled across the contexts.
  3. masked forward: every matrix outputs ((x @ R) * alpha) @ U^T on the masked model's own inputs.
  4. loss = mean over positions of KL(p_intact || p_masked). Gradients reach R and U through the masked
     forward and through alpha, including the threshold c's implicit dependence on every score.
  5. one Adam step, then exact-sum projection.

With temperature 1 in log-score space, alpha = a / (a + theta): no temperature knob.
MATRYOSHKA_FORWARD=hard runs step 3 with the exact hard top-k mask KLKeep uses and passes gradients to
the scores through the sigmoid top-k Jacobian (MAttr's "+hard" ablation); soft uses alpha both ways.
Exit code 75 = checkpoint saved at the time budget; the SLURM script resubmits.

Speed (same math as one micro-batch at a time): steps 1-4 run CHUNK contexts at a time with one budget draw
per BUDGET_CONTEXTS contexts; the bisection for c is one Triton kernel per chunk; the hard threshold comes from
topk, not a full sort; KL(p_intact || .) reuses the log-probs of the score cache's own forward (Fisher). With
CHUNK > MICRO the Fisher labels are sampled per CHUNK contexts, so they differ from the fisher_k4_5000 control's samples.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

from common import ROOT, HERE, MODULES, TARGET_SHA, CONTEXT, BATCH, credentials, tokens, batch, target, checkpoint, save_json
from train_pair import core, method
from gim_attribution import RULES, capture_gim
from fisher_attribution import capture_fisher

FISHER_K = 4
MICRO = 4                                                       # the earlier runs' micro-batch: budget draw and Fisher label chunk
CHUNK = int(os.environ.get('MATRYOSHKA_CHUNK_CONTEXTS', '16'))  # contexts per forward/backward; does not change the loss
K_MIN = int(os.environ.get('MATRYOSHKA_K_MIN', '32'))
BUDGET_CONTEXTS = int(os.environ.get('MATRYOSHKA_BUDGET_CONTEXTS', str(MICRO)))  # contexts sharing one budget draw
BUDGET_POWER = float(os.environ.get('MATRYOSHKA_BUDGET_POWER', '1'))  # budget density proportional to k^-power; 1 = log-uniform
BUDGET_STRATIFIED = os.environ.get('MATRYOSHKA_BUDGET_STRATIFIED', '0') == '1'
assert BUDGET_POWER > 0 and math.isfinite(BUDGET_POWER) and CHUNK % BUDGET_CONTEXTS == 0
FORWARD = os.environ.get('MATRYOSHKA_FORWARD', 'soft')
assert FORWARD in ('soft', 'hard')
SENSOR = os.environ.get('MATRYOSHKA_SENSOR', 'fisher')          # the per-token score that ranks pieces
assert SENSOR in ('fisher', 'gim')
LP_COEFF = float(os.environ.get('MATRYOSHKA_LP_COEFF', '0'))    # weight of the old par-auto Lp term; 0 = off
assert LP_COEFF >= 0 and math.isfinite(LP_COEFF)
ADV_COEFF = float(os.environ.get('MATRYOSHKA_ADV_COEFF', '0'))  # weight of VPD's persistent PGD loss (Adversary); 0 = off
assert ADV_COEFF >= 0 and math.isfinite(ADV_COEFF)
ADV_GATE = os.environ.get('MATRYOSHKA_ADV_GATE', 'hard')        # the adversary's keep level: hard = the forward's mask, soft = alpha
assert ADV_GATE in ('hard', 'soft')
MASS_COEFF = float(os.environ.get('MATRYOSHKA_MASS_COEFF', '0'))  # weight of the piece-mass penalty (mass_penalty); 0 = off
assert MASS_COEFF >= 0 and math.isfinite(MASS_COEFF)
SCORES = dict(fisher=f'|r.h| * RMS over K={FISHER_K} sampled next tokens of (g_k.u); plain CE gradients (fisher_k4_5000 sensor)',
              gim='|r.h| * |g.u|, g the GIM cotangent of the true next-token CE (gim_1h sensor: T=2 softmax, Q/K/V scaling, frozen RMS)')
TARGET_STEPS = int(os.environ.get('MATRYOSHKA_STEPS', '5000'))
MONITOR_EVERY = int(os.environ.get('MATRYOSHKA_MONITOR_EVERY', '100'))
MONITOR_BUDGETS = (43, 129, 430, 1290, 4301)       # 0.1%, 0.3%, 1%, 3%, 10% of 43,008
MONITOR_ROWS = slice(1000, 1004)                   # held-out; KLKeep uses val rows 0-127
CHECKPOINT_EVERY = 500
EXPERIMENT = 'matryoshka67_20260923'
OUT = Path(os.environ.get('MATRYOSHKA_OUT', ROOT/f'output/{EXPERIMENT}/main'))
COUNTS = [dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)[n.split('.')[-1]] for n in MODULES]
EXIT_CONTINUE = 75


if triton is not None:
    @triton.jit
    def _bisection_kernel(s_ptr, k_ptr, lo_ptr, hi_ptr, c_ptr, n_cols, row_stride,
                          ITERATIONS: tl.constexpr, BLOCK: tl.constexpr):
        """One program per row: the eager bisection below, with each step's sum of sigmoids in one pass over the row."""
        row = tl.program_id(0)
        base = s_ptr + row.to(tl.int64) * row_stride
        k = tl.load(k_ptr + row)
        lo = tl.load(lo_ptr + row)
        hi = tl.load(hi_ptr + row)
        for _ in range(ITERATIONS):
            mid = (lo + hi) / 2
            total = tl.zeros([BLOCK], dtype=tl.float32)
            for start in range(0, n_cols, BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                total += tl.sigmoid(tl.load(base + offsets, mask=offsets < n_cols, other=float('-inf')) + mid)
            over = tl.sum(total, axis=0) > k
            hi = tl.where(over, mid, hi)
            lo = tl.where(over, lo, mid)
        tl.store(c_ptr + row, (lo + hi) / 2)


def threshold_shift(s, k, iterations=40):
    """c per row, by bisection, so that sum_j sigmoid(s_j + c) = k; no gradient. k: a number or one value per row.
    Returns [n, 1]. CUDA float32 rows use the Triton kernel; anything else the eager loop (same steps)."""
    with torch.no_grad():
        k = torch.as_tensor(k, dtype=s.dtype, device=s.device).expand(s.shape[0]).contiguous()
        lo = -s.amax(-1) - 50.      # row sum ~ 0
        hi = -s.amin(-1) + 50.      # row sum ~ N
        if triton is not None and s.is_cuda and s.dtype == torch.float32:
            s = s.contiguous()
            c = torch.empty_like(lo)
            _bisection_kernel[(s.shape[0],)](s, k, lo, hi, c, s.shape[1], s.stride(0),
                                             ITERATIONS=iterations, BLOCK=4096, num_warps=8)
            return c[:, None]
        for _ in range(iterations):
            mid = (lo + hi) / 2
            over = torch.sigmoid(s + mid[:, None]).sum(-1) > k
            hi, lo = torch.where(over, mid, hi), torch.where(over, lo, mid)
        return ((lo + hi) / 2)[:, None]


class SigmoidTopk(torch.autograd.Function):
    """alpha = sigmoid(s + c) at the bisection's c. The gradient includes dc/ds_j = -sigma'_j / sum_i sigma'_i
    (implicit function theorem), so d alpha / d s = diag(sigma') - sigma' sigma'^T / sum(sigma'), the sigmoid
    top-k Jacobian: ds = sigma' (g - sum(sigma' g) / sum(sigma'))."""

    @staticmethod
    def forward(ctx, s, c):
        alpha = torch.sigmoid(s + c)
        ctx.save_for_backward(alpha)
        return alpha

    @staticmethod
    def backward(ctx, g):
        alpha, = ctx.saved_tensors
        slope = alpha * (1 - alpha)
        return slope * (g - (slope * g).sum(-1, keepdim=True) / slope.sum(-1, keepdim=True).clamp_min(1e-30)), None


class HardTopk(torch.autograd.Function):
    """Forward: the exact 0/1 top-k mask, s >= the row's k-th largest score (ties kept).
    Backward: the soft mask's gradient, passed straight through (MAttr's "+hard")."""

    @staticmethod
    def forward(ctx, soft, s, threshold):
        return (s >= threshold).to(soft.dtype)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def sigmoid_topk(s, k, iterations=40):
    """alpha = sigmoid(s + c), c solved per row by bisection so each row sums to k, with the implicit gradient."""
    return SigmoidTopk.apply(s, threshold_shift(s.detach(), k, iterations))


def forward_mask(s, ks, forward, group=None):
    """The mask used in the masked forward, and the soft sigmoid top-k mask (for statistics).
    ks: one budget per consecutive group of `group` rows (default: one budget for all rows)."""
    ks = [ks] if isinstance(ks, int) else list(ks)
    group = group or s.shape[0]
    assert len(ks) * group == s.shape[0], (len(ks), group, s.shape)
    per_row = torch.tensor(ks, dtype=s.dtype, device=s.device).repeat_interleave(group)
    soft = sigmoid_topk(s, per_row)
    if forward == 'soft': return soft, soft
    with torch.no_grad():
        scores = s.detach()
        threshold = torch.cat([scores[i*group:(i+1)*group].topk(k, dim=-1, sorted=False).values.amin(-1, keepdim=True)
                               for i, k in enumerate(ks)])
    return HardTopk.apply(soft, scores, threshold), soft


def capture(model, rows, seed, sensor=None, chunk=None, keep_logp=False):
    """Intact-model cache for the score: Fisher (K sampled-label backwards) or GIM (one modified backward).
    chunk (Fisher): contexts per forward/backward; the per-token math is the same, the sampled labels are not.
    keep_logp (Fisher) also returns the intact next-token log-probs [len(rows)*CONTEXT, V] from the cache's own
    forward, the same bf16-autocast forward as intact_logp."""
    if (sensor or SENSOR) == 'gim':
        assert not keep_logp
        return capture_gim(model, rows)
    if not keep_logp: return capture_fisher(model, rows, k=FISHER_K, seed=seed, microbatch=chunk or CHUNK)
    logps = []
    hook = model.register_forward_hook(lambda _m, _i, out: logps.append(out[0].detach().float().log_softmax(-1).flatten(0, 1)))
    try:
        cache = capture_fisher(model, rows, k=FISHER_K, seed=seed, microbatch=chunk or CHUNK)
    finally:
        hook.remove()
    return cache, torch.cat(logps)


def log_scores(cache, pieces, lo, hi, parts_out=None):
    """log score for positions lo:hi, all matrices concatenated in MODULES order: [n, N] fp32.
    parts_out (a list) also receives the per-matrix scores themselves, for the Lp term."""
    parts = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for entry, p in zip(cache, pieces):
            parts.append(method.structured.scores(method.structured.slice_entry(entry, slice(lo, hi)), p))
    if parts_out is not None: parts_out.extend(parts)
    return torch.cat(parts, -1).float().clamp_min(1e-30).log()


def lp_term(parts, lp):
    """Old par-auto concentration on the same scores: mean over matrices of Lp_m / Lp_m(initial pieces)."""
    return sum(method.lp_concentration(a.float(), lp['p'], lp['epsilon']).mean() / d
               for a, d in zip(parts, lp['denominators'], strict=True)) / len(parts)


def piece_mass(pieces, weight_norms):
    """Per matrix, the sum over its pieces of ||u_c||^2 ||r_c||^2 over ||W||_F^2 (weight_mass67.py's share): 1 when the
    pieces add up to the weights without cancelling, above 1 when large pieces cancel each other. [24]"""
    return torch.stack([(p.read.float().pow(2).sum(0) * p.write.float().pow(2).sum(0)).sum() / n for p, n in zip(pieces, weight_norms)])


def mass_penalty(pieces, weight_norms):
    """Mean over the 24 matrices of (piece mass - 1)^2: pulls every matrix's pieces toward adding up to its weights'
    mass. Both directions: below 1 means pieces split into overlapping copies."""
    return (piece_mass(pieces, weight_norms) - 1).pow(2).mean()


def masked_logits(model, ids, pieces, alphas):
    """Same replacement as evaluate_checkpoint.py's KLKeep hooks, with fractional masks allowed."""
    hooks = []
    try:
        for name, p, a in zip(MODULES, pieces, alphas):
            def replace(layer, inputs, output, p=p, a=a):
                return ((inputs[0] @ p.read) * a) @ p.write.T
            hooks.append(model.get_submodule(name).register_forward_hook(replace))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(ids)[0]
    finally:
        for hook in hooks: hook.remove()


def intact_logp(model, ids):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return model(ids)[0].float().log_softmax(-1)


def kl_per_position(target_logp, logits):
    return (target_logp.exp() * (target_logp - logits.float().log_softmax(-1))).sum(-1)


def vpd_adversary_config():
    """The PersistentPGDReconLoss entry of vpd_published.yaml (the config our paired VPD run trains with), checked
    against what Adversary implements."""
    import yaml
    raw = yaml.safe_load((HERE/'vpd_published.yaml').read_text())
    cfg, = [c for c in raw['loss_metric_configs'] if c['classname'] == 'PersistentPGDReconLoss']
    assert cfg['optimizer']['type'] == 'adam' and cfg['optimizer']['lr_schedule']['fn_type'] == 'constant', cfg
    assert cfg['scope'] == dict(type='per_batch_per_position') and not cfg['use_sigmoid_parameterization'], cfg
    assert cfg.get('n_samples', 1) == 1 and cfg.get('start_frac', 0.) == 0., cfg
    return cfg


class Adversary:
    """VPD's persistent PGD loss (PersistentPGDReconLoss in vpd_published.yaml, as our paired VPD run trains it) with
    the hard top-k keep set at each context's budget in place of VPD's gate ci: mask = keep + (1 - keep) * source,
    on all 24 matrices. One source in [0, 1] per context slot, position and piece (VPD's per_batch_per_position scope),
    uniform at the start and kept across updates. Each update, in param_decomp.persistent_pgd's order: n_warmup_steps
    ascent steps on this batch, then the KL at the updated sources enters the loss (weight coeff; the pieces and their
    ranking train through it), then one more ascent step with that KL's gradient. An ascent step is VPD's
    AdamPGDOptimizer.step on the gradient of the batch-mean KL, then clamping to [0, 1]. There is no source for
    a delta component: the pieces sum to the weights exactly."""

    def __init__(self, cfg, coeff, contexts, seed):
        optimizer = cfg['optimizer']
        self.coeff, self.warmup, self.schedule = coeff, cfg['n_warmup_steps'], optimizer['lr_schedule']
        self.beta1, self.beta2, self.eps = optimizer['beta1'], optimizer['beta2'], optimizer['eps']
        generator = torch.Generator(device='cuda').manual_seed(seed)
        self.source = torch.rand(contexts, CONTEXT, sum(COUNTS), device='cuda', generator=generator)
        self.m, self.v = torch.zeros_like(self.source), torch.zeros_like(self.source)
        self.steps = 0              # ascent steps so far: n_warmup_steps + 1 per update
        self.lr = 0.
        self.gradients = None       # tests: a list that receives (rows, step number, gradient) of every ascent step

    def set_lr(self, step, total):
        """param_decomp's get_scheduled_value for a constant schedule with linear warmup."""
        warmup = int(total * self.schedule['warmup_pct'])
        self.lr = self.schedule['start_val'] * (step / warmup) if step < warmup else self.schedule['start_val']

    def ascend(self, rows, grad, number):
        """AdamPGDOptimizer.step on the sources of context slots `rows`, as the batch's ascent step `number`, then clamp."""
        if self.gradients is not None: self.gradients.append((rows, number, grad.detach().clone()))
        with torch.no_grad():
            m, v, source = self.m[rows], self.v[rows], self.source[rows]
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            m_hat = m / (1 - self.beta1 ** number)
            v_hat = v / (1 - self.beta2 ** number)
            source.add_(self.lr * m_hat / v_hat.sqrt().add_(self.eps))
            source.clamp_(0., 1.)

    @staticmethod
    def masks(keep, source, ids):
        mixed = keep + (1 - keep) * source.flatten(0, 1)        # VPD's _interpolate_component_mask
        return [a.reshape(*ids.shape, -1) for a in mixed.split(COUNTS, -1)]

    def kl(self, model, ids, target, pieces, keep, rows, positions, groups):
        """The warmup ascent steps on the sources of context slots `rows`, then the KL per budget group at the updated
        sources, with gradients to the pieces, the keep set and the returned sources leaf (for finish). Also returns
        the KL per group at the sources as they came in. positions: the whole batch's, as VPD's mean."""
        gate = keep.detach()
        frozen = [core.FreePieces(p.read.detach(), p.write.detach()) for p in pieces]
        start = None
        for j in range(self.warmup):
            source = self.source[rows].detach().requires_grad_()
            kl = kl_per_position(target, masked_logits(model, ids, frozen, self.masks(gate, source, ids)))
            if start is None: start = kl.detach().view(groups, -1).mean(-1)
            grad, = torch.autograd.grad(kl.sum() / positions, source)
            self.ascend(rows, grad, self.steps + j + 1)
            del kl, grad
        source = self.source[rows].detach().requires_grad_()
        kl = kl_per_position(target, masked_logits(model, ids, pieces, self.masks(keep, source, ids))).view(groups, -1).mean(-1)
        return kl, (kl.detach() if start is None else start), source

    def finish(self, rows, source):
        """The update's last ascent step on these contexts: source.grad holds coeff times the batch-mean KL's gradient."""
        self.ascend(rows, source.grad / self.coeff, self.steps + self.warmup + 1)

    def state(self):
        return dict(source=self.source, m=self.m, v=self.v, steps=self.steps)

    def load(self, state):
        with torch.no_grad():
            for name in ('source', 'm', 'v'): getattr(self, name).copy_(state[name])
        self.steps = state['steps']


def budget_at(u, total):
    """The budget at quantile u of the density proportional to k^-BUDGET_POWER on [K_MIN, total - 1].
    Power 1 is log-uniform (the earlier runs); above 1, small budgets are drawn more often."""
    lo, hi = K_MIN, total - 1
    if BUDGET_POWER == 1:
        k = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        e = 1 - BUDGET_POWER
        k = (lo ** e + u * (hi ** e - lo ** e)) ** (1 / e)
    return min(total - 1, max(1, round(k)))


def budget(generator, total):
    return budget_at(float(torch.rand((), generator=generator)), total)


def update_budgets(generator, total, count):
    """The update's `count` budgets, one per BUDGET_CONTEXTS contexts. Independent draws, or (BUDGET_STRATIFIED) one
    quantile from each of `count` equal-probability slices, shuffled across the contexts: every update then covers
    the whole distribution, and each context's budget still has the distribution of an independent draw."""
    if not BUDGET_STRATIFIED:
        return [budget(generator, total) for _ in range(count)]
    quantiles = (torch.arange(count, dtype=torch.float64) + torch.rand(count, generator=generator, dtype=torch.float64)) / count
    return [budget_at(float(u), total) for u in quantiles[torch.randperm(count, generator=generator)]]


def accumulate(model, rows, cache, feasible, draws, forward=FORWARD, lp=None, target_logp=None, chunk=None, adversary=None):
    """Masked-forward KL over the update's budget groups (len(rows) // len(draws) consecutive contexts share each
    budget draw), run `chunk` contexts at a time; gradients land on the feasible factors' leaves. The loss is the same
    as one group at a time: the sum over groups of (mean KL + lp coeff * lp_term) / their number.
    lp (dict coeff, p, epsilon, denominators) adds coeff * lp_term per group (the chunk's term is the mean of its
    groups' terms). target_logp: intact log-probs of all rows [len(rows)*CONTEXT, V], or None to recompute them.
    adversary (an Adversary for these rows' context slots) adds coeff * its KL per group, at the group's keep set,
    and makes the update's ascent steps on its sources.

    Returns the leaves (with .grad) and per-group stats."""
    chunk = chunk or CHUNK
    per = len(rows) // len(draws)
    assert per * len(draws) == len(rows) and chunk % per == 0, (len(rows), len(draws), chunk)
    assert adversary is None or adversary.source.shape[0] == len(rows), 'one adversary source slot per context'
    leaves = [core.FreePieces(f.read.detach().requires_grad_(), f.write.detach().requires_grad_()) for f in feasible]
    count = len(draws)
    pending = []
    for first in range(0, len(rows), chunk):
        block = rows[first:first+chunk]
        ids = block[:, :-1]
        lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
        ks = draws[first//per:(first+len(block))//per]
        target = intact_logp(model, ids) if target_logp is None else target_logp[lo:hi].view(*ids.shape, -1)
        parts = [] if lp else None
        s = log_scores(cache, leaves, lo, hi, parts_out=parts)
        mask, soft = forward_mask(s, ks, forward, per*CONTEXT)
        alphas = [a.reshape(*ids.shape, -1) for a in mask.split(COUNTS, -1)]
        kl = kl_per_position(target, masked_logits(model, ids, leaves, alphas)).view(len(ks), -1).mean(-1)   # per budget group
        term = lp_term(parts, lp) if lp else None
        loss = kl.sum() if term is None else kl.sum() + lp['coeff'] * term * len(ks)
        adv = None
        if adversary is not None:
            here = slice(first, first + len(block))
            adv_kl, adv_start, source = adversary.kl(model, ids, target, leaves, soft if ADV_GATE == 'soft' else mask, here, len(rows)*CONTEXT, len(ks))
            loss = loss + adversary.coeff * adv_kl.sum()
            adv = (adv_kl.detach(), adv_start)
        (loss / count).backward()
        if adversary is not None:
            adversary.finish(here, source)
            del adv_kl, adv_start, source
        with torch.no_grad():
            on = soft > .5
            pending.append((ks, kl.detach(), on.sum(-1).float().view(len(ks), -1).mean(-1),
                            (soft * ~on).sum(-1).view(len(ks), -1).mean(-1), None if term is None else term.detach(), adv))
        del target, s, mask, soft, alphas, kl, on, parts, term, loss, adv
    if adversary is not None: adversary.steps += adversary.warmup + 1
    stats = []                                       # one host sync for the whole update
    for ks, kl, on, leak, term, adv in pending:
        kl, on, leak = kl.tolist(), on.tolist(), leak.tolist()
        term = None if term is None else float(term)
        adv = None if adv is None else [a.tolist() for a in adv]
        for j, k in enumerate(ks):
            stats.append(dict(k=k, kl=kl[j], on_over_k=on[j] / k, soft_leak=leak[j] / k))
            if term is not None: stats[-1]['lp_normalized'] = term
            if adv is not None: stats[-1].update(adv_kl=adv[0][j], adv_kl_start=adv[1][j])
    return leaves, stats


def monitor(model, rows, pieces, seed):
    """Hard top-k KL (KLKeep's mask, on 4 held-out contexts, the run's own ranking) at fixed budgets."""
    pieces = [core.FreePieces(p.read.detach(), p.write.detach()) for p in pieces]
    cache = capture(model, rows, seed)      # needs grad mode for its backwards
    ids = rows[:, :-1]
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scores = torch.cat([method.structured.scores(e, p) for e, p in zip(cache, pieces)], -1).float()
        del cache
        order = scores.argsort(dim=-1, descending=True, stable=True)
        target_logp = intact_logp(model, ids)
        result = {}
        for k in MONITOR_BUDGETS + (scores.shape[-1],):
            keep = torch.zeros_like(scores).scatter_(-1, order[:, :k], 1.)
            alphas = [m.reshape(*ids.shape, -1) for m in keep.split(COUNTS, -1)]
            result[k] = float(kl_per_position(target_logp, masked_logits(model, ids, pieces, alphas)).mean())
    return result


def main():
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    resuming = (OUT/'latest_checkpoint.json').exists()
    if (OUT/'plan.json').exists() and not resuming:
        raise FileExistsError(f'{OUT} has a plan but no checkpoint; remove it to restart')
    deadline = float(os.environ['MATRYOSHKA_DEADLINE'])
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGUSR1, request_stop)
    import wandb
    data, manifest = tokens(); model = target()
    val, val_manifest = tokens('val')
    monitor_rows = torch.from_numpy(np.array(val[MONITOR_ROWS], dtype=np.int64)).cuda()
    weights = [model.get_submodule(n).weight for n in MODULES]
    total = sum(COUNTS)
    control = json.loads((ROOT/'output/gim_1h/plan.json').read_text())
    assert control['sensor'] == 'gim' and control['rules'] == RULES
    assert control['target_sha256'] == TARGET_SHA and control['data_sha256'] == manifest['train']['sha256']
    assert control['component_counts'] == dict(zip(MODULES, COUNTS))
    cfg = method.TrainConfig(**control['training_config'])
    assert cfg.steps == 8700 and cfg.precision == 'bf16' and cfg.projection_precision == 'fp32'
    assert BATCH % BUDGET_CONTEXTS == 0
    plan = dict(method=f'matryoshka: sigmoid top-k over the {SENSOR} read x write score, masked-forward KL',
                sensor=SENSOR, score=SCORES[SENSOR],
                mask='alpha = sigmoid(log a + c), c per token so sum alpha = k; temperature 1, i.e. alpha = a/(a+theta)',
                forward=FORWARD, forward_note='soft: alpha in the masked forward; hard: exact top-k forward, sigmoid top-k Jacobian backward',
                threshold_gradient='implicit: dc/ds_j = -sigma_j\' / sum sigma\'',
                budget=dict(distribution='log-uniform' if BUDGET_POWER == 1 else f'density proportional to k^-{BUDGET_POWER:g}',
                            k_min=K_MIN, k_max=total-1, contexts_per_draw=BUDGET_CONTEXTS,
                            draws='stratified: one per equal-probability slice each update, shuffled' if BUDGET_STRATIFIED else 'independent'),
                loss='mean over positions of KL(p_intact || p_masked), masked forward replaces all 24 matrices',
                sparsity_penalty=None, gate_network=None, fisher_k=FISHER_K,
                training_config=asdict(cfg), lr_note='lr/lr_min/8700-step cosine horizon and Adam betas of fisher_k4_5000; p_end unused',
                stop_after=TARGET_STEPS, matched_control=dict(fisher='output/fisher_k4_5000 (KLKeep eval_fisher_k8)',
                                                              gim='output/gim_1h at 5000 (KLKeep GIM and Fisher K=8 rankings)')[SENSOR],
                initialization='same seed-0 projected factors as the control', data='same train order as the control',
                monitor=dict(val_rows=[MONITOR_ROWS.start, MONITOR_ROWS.stop], budgets=list(MONITOR_BUDGETS), every=MONITOR_EVERY,
                             ranking=f'{SENSOR} score, hard global top-k'),
                target_sha256=TARGET_SHA, data_sha256=manifest['train']['sha256'], val_sha256=val_manifest['val']['sha256'],
                component_counts=dict(zip(MODULES, COUNTS)), gpu=torch.cuda.get_device_name(),
                implementation=dict(chunk_contexts=CHUNK, bisection='triton kernel' if triton is not None else 'eager',
                                    hard_threshold='topk', kl_target='score cache forward' if SENSOR == 'fisher' else 'separate intact forward',
                                    fisher_labels=f'sampled per {CHUNK} contexts'))
    if LP_COEFF:
        plan['lp'] = dict(coeff=LP_COEFF, epsilon=cfg.lp_epsilon, p_end=cfg.p_end,
                          p_schedule='1 -> p_end over the first 80% of the 8,700-step horizon, as the control',
                          term='mean over the 24 matrices of sum_c ((a_c+eps)^p - eps^p), token-averaged, over its value for the '
                               'initial pieces on the control\'s fixed 256-position reference set (seed 123, first batch), at the current p',
                          scores='the same per-token scores that rank the pieces, on the intact cache')
    if ADV_COEFF:
        adversary_cfg = vpd_adversary_config()
        plan['adversary'] = dict(
            coeff=ADV_COEFF, coeff_note='VPD weights this loss equally with its stochastic masked-output loss (0.5 each); '
                                        'here it has the masked-forward KL\'s weight',
            vpd_config=adversary_cfg, vpd_source='vpd_published.yaml PersistentPGDReconLoss (param_decomp.persistent_pgd)',
            gate='the hard top-k keep set at the context\'s budget, as in the masked forward (VPD: ci.lower_leaky)',
            mask='keep + (1 - keep) * source on all 24 matrices (VPD: AllLayersRouter)',
            sources=f'[{BATCH}, {CONTEXT}, {total}] fp32, uniform [0, 1] (cuda generator seed {cfg.seed + 606}), '
                    'kept across updates; no delta-component source',
            update=f'{adversary_cfg["n_warmup_steps"]} ascent steps, then the loss at the updated sources, then one more ascent '
                   'step with its gradient; each an AdamPGDOptimizer step on the batch-mean KL, clamped to [0, 1]',
            lr_schedule_horizon=cfg.steps,
            loss='coeff * mean over positions of KL(p_intact || p_adversarial), added to the masked-forward loss')
        if ADV_GATE == 'soft':
            plan['adversary']['gate'] = ('the sigmoid top-k alpha at the context\'s budget (soft, in [0, 1], with its own gradient); '
                                         'the masked forward keeps the hard top-k mask')
    if MASS_COEFF:
        plan['mass'] = dict(coeff=MASS_COEFF, term='mean over the 24 matrices of (sum_c ||u_c||^2 ||r_c||^2 / ||W||_F^2 - 1)^2, '
                                                   'on the projected pieces, once per update, gradient through the projection')
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, COUNTS)]
    weight_mass = float(sum(w.double().pow(2).sum() for w in weights))
    weight_norms = [float(w.double().pow(2).sum()) for w in weights]
    initial_scores = None
    if LP_COEFF:
        # Old par-auto normalization: each matrix's Lp divided by its value for the initial pieces on a fixed set.
        index = torch.randperm(BATCH*CONTEXT, device='cuda', generator=torch.Generator(device='cuda').manual_seed(123))[:256]
        first = capture(model, batch(data, 0), seed=123, chunk=MICRO)     # the earlier runs' labels, so the same normalization
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            initial_scores = [method.structured.scores(method.structured.slice_entry(e, index), p).float() for e, p in zip(first, pieces)]
        del first
    parameters = [t.requires_grad_() for p in pieces for t in (p.read, p.write)]
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr, betas=(0.9, 0.99))
    budgets = torch.Generator().manual_seed(cfg.seed + 505)
    adversary = Adversary(adversary_cfg, ADV_COEFF, BATCH, seed=cfg.seed + 606) if ADV_COEFF else None
    start = 0
    if resuming:
        saved_plan = json.loads((OUT/'plan.json').read_text())
        assert {k: v for k, v in saved_plan.items() if k not in ('gpu', 'job_ids')} == json.loads(json.dumps({k: v for k, v in plan.items() if k != 'gpu'}))
        full = torch.load(json.loads((OUT/'latest_checkpoint.json').read_text())['full_checkpoint'], map_location='cpu', weights_only=False)
        with torch.no_grad():
            for p, s in zip(pieces, full['state']['pieces'], strict=True):
                p.read.copy_(s['read']); p.write.copy_(s['write'])
        optimizer.load_state_dict(full['state']['optimizer'])
        budgets.set_state(full['state']['budgets'])
        if adversary is not None: adversary.load(full['state']['adversary'])
        torch.set_rng_state(full['torch_rng']); torch.cuda.set_rng_state(full['cuda_rng'])
        start = full['completed']
        del full
        print(json.dumps(dict(resumed_from=start)), flush=True)
    else:
        save_json(plan, OUT/'plan.json'); print(json.dumps(dict(plan=plan)), flush=True)
    previous = json.loads((OUT/'wandb.json').read_text()) if resuming else None
    run = wandb.init(project='matryoshka', entity='rohan-kathuria-neu', group=EXPERIMENT,
                     name=f'parauto-matryoshka-{FORWARD}-fisher-k4-5000' if OUT.name == 'main' else f'parauto-matryoshka-{OUT.name}',
                     job_type='training', dir=str(OUT), config=plan,
                     id=previous['id'] if previous else None, resume='must' if previous else None)
    save_json(dict(id=run.id, url=run.url), OUT/'wandb.json')
    run.define_metric('update'); run.define_metric('train/*', step_metric='update'); run.define_metric('monitor/*', step_metric='update')

    def save(completed):
        before = time.perf_counter()
        out = OUT/'checkpoints'/f'{completed:06d}'
        identity = dict(method='parauto', sensor=SENSOR, completed=completed, training_positions=completed*BATCH*CONTEXT,
                        data_sha256=manifest['train']['sha256'], target_sha256=TARGET_SHA)
        factors = [{'read': p.read.detach(), 'write': p.write.detach()} for p in pieces]
        checkpoint(dict(**identity, factors=factors, provenance=dict(kind='Matryoshka masked-forward training', plan=plan,
                        training_config=asdict(cfg))), out/'model.pth')
        state = dict(pieces=factors, optimizer=optimizer.state_dict(), budgets=budgets.get_state())
        if adversary is not None: state['adversary'] = adversary.state()
        checkpoint(dict(**identity, state=state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), out/'checkpoint.pt')
        old = json.loads((OUT/'latest_checkpoint.json').read_text()) if (OUT/'latest_checkpoint.json').exists() else None
        record = dict(step=completed, full_checkpoint=str(out/'checkpoint.pt'), model=str(out/'model.pth'), seconds=time.perf_counter()-before)
        save_json(record, OUT/'latest_checkpoint.json')
        # Keep every model.pth, but only the newest full (optimizer) checkpoint.
        if old and old['full_checkpoint'] != record['full_checkpoint']: Path(old['full_checkpoint']).unlink(missing_ok=True)
        print(json.dumps(dict(checkpoint=record)), flush=True)
        run.log({'update': completed, 'train/checkpoint_seconds': record['seconds']})

    def log_monitor(completed):
        before = time.perf_counter()
        curve = monitor(model, monitor_rows, pieces, seed=77)
        row = {f'monitor/kl_keep_{k}': v for k, v in curve.items()}
        run.log({'update': completed, **row, 'monitor/seconds': time.perf_counter()-before})
        print(json.dumps(dict(update=completed, monitor=curve)), flush=True)

    if start == 0: log_monitor(0)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    completed = start
    for step in range(start, TARGET_STEPS):
        tick = time.perf_counter()
        for group in optimizer.param_groups: group['lr'] = core.cosine_lr(step, cfg)
        if adversary is not None: adversary.set_lr(step, cfg.steps)
        optimizer.zero_grad(set_to_none=True)
        rows = batch(data, step)
        # The control's seeds, step for step (its labels too when CHUNK == MICRO).
        cache, target_logp = capture(model, rows, seed=1_000_003+step, keep_logp=True) if SENSOR == 'fisher' else \
                             (capture(model, rows, seed=1_000_003+step), None)
        feasible = [method.project(w, p, cfg) for w, p in zip(weights, pieces)]
        draws = update_budgets(budgets, total, BATCH // BUDGET_CONTEXTS)
        lp = None
        if LP_COEFF:
            p_now = 1 + (cfg.p_end - 1) * min(1.0, step / max(1, cfg.steps * 0.8))        # method.fit's schedule
            lp = dict(coeff=LP_COEFF, p=p_now, epsilon=cfg.lp_epsilon,
                      denominators=[method.lp_concentration(s, p_now, cfg.lp_epsilon).mean().clamp_min(1e-8) for s in initial_scores])
        leaves, stats = accumulate(model, rows, cache, feasible, draws, lp=lp, target_logp=target_logp, adversary=adversary)
        del cache, target_logp
        # Push the leaf gradients through the in-graph projection onto the free factors, as method.fit does.
        outputs, cotangents = [t for f in feasible for t in (f.read, f.write)], [t.grad for l in leaves for t in (l.read, l.write)]
        if MASS_COEFF:
            masses = piece_mass(feasible, weight_norms)
            outputs.append(MASS_COEFF * (masses - 1).pow(2).mean()); cotangents.append(torch.ones((), device=masses.device))
        torch.autograd.backward(outputs, cotangents)
        del leaves, feasible
        grad_norm = float(torch.linalg.vector_norm(torch.stack([t.grad.norm() for t in parameters])))
        if not math.isfinite(grad_norm): raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        for w, p in zip(weights, pieces): method.project_(w, p, cfg)
        completed = step + 1
        torch.cuda.synchronize()
        kls = [s['kl'] for s in stats]
        row = dict(kl=float(np.mean(kls)), lr=core.cosine_lr(step, cfg), grad_norm=grad_norm,
                   on_over_k=float(np.mean([s['on_over_k'] for s in stats])),
                   soft_leak=float(np.mean([s['soft_leak'] for s in stats])),
                   seconds_this_step=time.perf_counter()-tick,
                   seconds_per_step=(time.perf_counter()-started)/(completed-start),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), positions=completed*BATCH*CONTEXT)
        if lp:
            row.update(lp_normalized=float(np.mean([s['lp_normalized'] for s in stats])), p=lp['p'])
        if MASS_COEFF:
            m = masses.detach()
            row.update(mass_penalty=float((m - 1).pow(2).mean()), mass_max=float(m.max()),
                       mass_total=float((m * torch.tensor(weight_norms, device=m.device)).sum()) / weight_mass)
        if adversary is not None:
            row.update(adv_kl=float(np.mean([s['adv_kl'] for s in stats])), adv_kl_start=float(np.mean([s['adv_kl_start'] for s in stats])),
                       adv_lr=adversary.lr, adv_source_mean=float(adversary.source.mean()))
        for lo, hi, label in ((0, 100, 'lt100'), (100, 1000, '100_1000'), (1000, 10000, '1000_10000'), (10000, total, 'ge10000')):
            inside = [s for s in stats if lo <= s['k'] < hi]
            if inside: row[f'kl_k_{label}'] = float(np.mean([s['kl'] for s in inside]))
            if inside and adversary is not None: row[f'adv_kl_k_{label}'] = float(np.mean([s['adv_kl'] for s in inside]))
        if not all(math.isfinite(v) for v in row.values()): raise RuntimeError(f'Nonfinite metrics: {row}')
        run.log({'update': completed, **{'train/'+k: v for k, v in row.items()}})
        if completed % 10 == 0 or completed <= 5:
            extra = {} if adversary is None else dict(adv_kl_by_budget=[s['adv_kl'] for s in stats])
            print(json.dumps(dict(update=completed, **row, budgets=draws, kl_by_budget=kls, **extra)), flush=True)
        if completed % 100 == 0 or completed == TARGET_STEPS:
            errors = [core.reconstruction_error(w, p) for w, p in zip(weights, pieces)]
            # Sum of piece sizes ||u||^2 ||r||^2 over the weights' ||W||^2: 1 = no cancellation between pieces.
            with torch.no_grad():
                total_mass = sum(float((p.read.double().pow(2).sum(0) * p.write.double().pow(2).sum(0)).sum()) for p in pieces) / weight_mass
            run.log({'update': completed, 'train/max_sum_error': max(errors), 'train/piece_mass_total': total_mass})
            if max(errors) > 1e-5: raise RuntimeError(f'Matrix sum drift: {max(errors)}')
            save_json(dict(update=completed, **row), OUT/'progress.json')
        if completed % MONITOR_EVERY == 0 or completed == TARGET_STEPS: log_monitor(completed)
        out_of_time = stop or time.time() >= deadline
        if completed % CHECKPOINT_EVERY == 0 or completed == TARGET_STEPS or out_of_time: save(completed)
        if out_of_time and completed < TARGET_STEPS:
            run.finish(); print(json.dumps(dict(paused_at=completed, reason='signal' if stop else 'time_budget')), flush=True)
            sys.exit(EXIT_CONTINUE)
    result = dict(completed=completed, positions=completed*BATCH*CONTEXT, seconds_this_job=time.perf_counter()-started,
                  resumed_from=start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  data_sha256=manifest['train']['sha256'], target_sha256=TARGET_SHA)
    save_json(result, OUT/'result.json'); run.summary.update(result); run.finish()
    print(json.dumps(dict(finished=result)), flush=True)


# Paired controls on the same 65,536 held-out positions: (our eval dir, control KLKeep, control label, output name).
COMPARISONS = dict(
    fisher=[('eval_fisher_k8', 'output/fisher_k4_5000/eval_fisher_k8/klkeep.json', 'fisher_k4_5000', 'comparison.json')],
    gim=[('eval_gim', 'output/evaluations/gim_normal_vpd_5000/gim/klkeep.json', 'gim_1h_5000', 'comparison_gim.json'),
         ('eval_fisher_k8', 'output/fisher_rank/gim_p05_fisher_k8/klkeep.json', 'gim_1h_5000', 'comparison.json')])


def publish_evaluation():
    """After KLKeep: write the paired comparison(s) to the matched control as JSON (not logged to W&B)."""
    for folder, path, label, name in COMPARISONS[SENSOR]:
        ours = json.loads((OUT/folder/'klkeep.json').read_text())
        control = json.loads((ROOT/path).read_text())
        assert ours['data_sha256'] == control['data_sha256'] and ours['positions'] == control['positions']
        assert ours['sensor'] == control['sensor'], (ours['sensor'], control['sensor'])
        rows = [dict(keep_fraction=a['keep_fraction'], kept_components=a['kept_components'], kl_matryoshka=a['kl'], **{f'kl_{label}': b['kl']})
                for a, b in zip(ours['curve'], control['curve'], strict=True) if a['keep_fraction'] == b['keep_fraction']]
        save_json(dict(rows=rows, ranking=ours['sensor'], control=path, positions=ours['positions']), OUT/name)
        print(json.dumps({name: rows}), flush=True)


if __name__ == '__main__':
    publish_evaluation() if sys.argv[1:] == ['publish'] else main()
