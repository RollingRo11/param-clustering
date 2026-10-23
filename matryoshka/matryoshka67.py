"""Matryoshka training of exact-sum rank-one pieces (after MAttr, arXiv 2609.25518), 5,000 steps.

par-auto's own read x write score (Fisher or GIM) picks which pieces each token keeps; a sigmoid top-k mask at a
random budget k keeps them; the loss is the KL of the actually masked model to the intact one. There is no gate
network; by default no Lp term and no PGD (MATRYOSHKA_LP_COEFF adds the old Lp term, MATRYOSHKA_ADV_COEFF adds VPD's
persistent PGD loss, see Adversary, MATRYOSHKA_MASS_COEFF a pull of each matrix's piece mass toward 100%, see mass_penalty,
MATRYOSHKA_FREQ_BETA prices each piece's Lp share by its usage, VPD's frequency term, see frequency_lp,
MATRYOSHKA_CERT_GAMMA a second-order bound on the KL of switching dropped pieces back on, see tail_certificate;
MATRYOSHKA_INIT starts a new run directory from another run's full checkpoint). Everything else (init, data order, score cache, lr schedule,
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

Mainline options (the 21 Oct 2026 recipe, main.md; all off by default):
  MATRYOSHKA_ADV_FRACTION=f   merged loss: each context is attacked with probability f (else plain); an attacked
                              context's only KL is at the attack mask keep + (1 - keep) * source (MergedAdversary:
                              per-token persistent sources, MATRYOSHKA_PARTICLES per slot, VPD param-decomp 177d0cc).
                              MATRYOSHKA_ADV_LR / _ADV_BETA1 / _ADV_BETA2 / _ADV_LR_WARMUP override the attacker's Adam.
  MATRYOSHKA_ROUTING=uniform_k_subset   per position, k ~ U{1..24} matrices masked, the rest run with W (VPD's
                              UniformKSubsetRouter, App. A.3.3); the same routing for every pass of an update's chunk.
  MATRYOSHKA_FREQ_COEFF       hard-set frequency term, see hard_frequency.
  MATRYOSHKA_TAIL_COEFF       share of each token's score outside its keep set, see tail_share.
  MATRYOSHKA_SCHEDULE_STEPS   horizon of the lr cosine, the p schedule and the attacker's lr warm-up (control: 8,700).
  MATRYOSHKA_PROJECTION=fp64  exact-sum projection in float64 (the 13B run's fix for slow fp32 drift).

MATRYOSHKA_TARGET=simplestories trains on the 2-layer SimpleStories target (gf6rbga0, 7,104 pieces; common_ss.py) instead
of the 67M one, with the same method and options; its training config is the control's, inlined (no gim_1h plan to read).
MATRYOSHKA_WANDB_DIR puts W&B's local files somewhere other than the run directory.

With temperature 1 in log-score space, alpha = a / (a + theta): no temperature knob.
MATRYOSHKA_FORWARD=hard runs step 3 with the exact hard top-k mask KLKeep uses and passes gradients to
the scores through the sigmoid top-k Jacobian (MAttr's "+hard" ablation); soft uses alpha both ways.
Exit code 75 = checkpoint saved at the time budget; the SLURM script resubmits.

Speed (same math as one micro-batch at a time): steps 1-4 run CHUNK contexts at a time with one budget draw
per BUDGET_CONTEXTS contexts; the bisection for c is one Triton kernel per chunk; the hard threshold comes from
topk, not a full sort; KL(p_intact || .) reuses the log-probs of the score cache's own forward (Fisher). With
CHUNK > MICRO the Fisher labels are sampled per CHUNK contexts, so they differ from the fisher_k4_5000 control's samples.
"""
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
# Old cluster snapshots hold only this file and import the rest from code/compare; a git checkout has it all beside it.
if (ROOT/'code/compare').is_dir() and not (Path(__file__).resolve().parent/'train_pair.py').exists():
    sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

from common import ROOT, HERE, MODULES, TARGET_SHA, CONTEXT, BATCH, PIECE_COUNTS, EXPERIMENT, TARGET, credentials, tokens, batch, target, checkpoint, save_json
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
FREQ_BETA = float(os.environ.get('MATRYOSHKA_FREQ_BETA', '0'))  # VPD's frequency term on the Lp term (frequency_lp); 0 = off
assert FREQ_BETA >= 0 and math.isfinite(FREQ_BETA) and (FREQ_BETA == 0 or LP_COEFF > 0), 'the frequency term prices the Lp term'
MASS_COEFF = float(os.environ.get('MATRYOSHKA_MASS_COEFF', '0'))  # weight of the piece-mass penalty (mass_penalty); 0 = off
assert MASS_COEFF >= 0 and math.isfinite(MASS_COEFF)
CERT_GAMMA = float(os.environ.get('MATRYOSHKA_CERT_GAMMA', '0'))  # weight of the signed tail certificate (tail_certificate); 0 = off
CERT_TAU = float(os.environ.get('MATRYOSHKA_CERT_TAU', '0.25'))    # nats of certificate allowed per token before the penalty starts
assert CERT_GAMMA >= 0 and math.isfinite(CERT_GAMMA) and CERT_TAU >= 0 and math.isfinite(CERT_TAU)
assert CERT_GAMMA == 0 or os.environ.get('MATRYOSHKA_SENSOR', 'fisher') == 'fisher', 'the certificate needs the K sampled Fisher cotangents'
INIT = os.environ.get('MATRYOSHKA_INIT')  # a full checkpoint.pt of another run to start a new run directory from (fine-tuning)
ADV_FRACTION = float(os.environ.get('MATRYOSHKA_ADV_FRACTION', '0'))  # share of contexts trained at the attack mask (MergedAdversary); 0 = off
assert 0 <= ADV_FRACTION <= 1 and (ADV_FRACTION == 0 or ADV_COEFF == 0), 'the merged loss replaces the added adversarial loss'
assert ADV_FRACTION == 0 or BUDGET_CONTEXTS == 1, 'the merged loss attacks whole budget groups: one context per budget draw'
PARTICLES = int(os.environ.get('MATRYOSHKA_PARTICLES', '1'))          # attacker sources per batch slot (VPD's particle pool)
assert PARTICLES >= 1
ADV_OVERRIDES = {k: float(os.environ[f'MATRYOSHKA_ADV_{e}']) for k, e in
                 (('lr', 'LR'), ('beta1', 'BETA1'), ('beta2', 'BETA2'), ('warmup_pct', 'LR_WARMUP')) if f'MATRYOSHKA_ADV_{e}' in os.environ}
ROUTING = os.environ.get('MATRYOSHKA_ROUTING', 'all')                 # which matrices a position masks; all = every one
assert ROUTING in ('all', 'uniform_k_subset')
FREQ_COEFF = float(os.environ.get('MATRYOSHKA_FREQ_COEFF', '0'))      # weight of the hard-set frequency term (hard_frequency); 0 = off
TAIL_COEFF = float(os.environ.get('MATRYOSHKA_TAIL_COEFF', '0'))      # weight of the tail-share term (tail_share); 0 = off
assert FREQ_COEFF >= 0 and math.isfinite(FREQ_COEFF) and TAIL_COEFF >= 0 and math.isfinite(TAIL_COEFF)
assert (FREQ_COEFF == 0 and TAIL_COEFF == 0) or FORWARD == 'hard', 'both terms are defined on the hard keep set'
SCHEDULE_STEPS = int(os.environ.get('MATRYOSHKA_SCHEDULE_STEPS', '8700'))  # lr cosine / p schedule / attacker warm-up horizon
PROJECTION = os.environ.get('MATRYOSHKA_PROJECTION', 'fp32')
assert SCHEDULE_STEPS > 1 and PROJECTION in ('fp32', 'fp64')
SCORES =dict(fisher=f'|r.h| * RMS over K={FISHER_K} sampled next tokens of (g_k.u); plain CE gradients (fisher_k4_5000 sensor)',
              gim='|r.h| * |g.u|, g the GIM cotangent of the true next-token CE (gim_1h sensor: T=2 softmax, Q/K/V scaling, frozen RMS)')
TARGET_STEPS = int(os.environ.get('MATRYOSHKA_STEPS', '5000'))
MONITOR_EVERY = int(os.environ.get('MATRYOSHKA_MONITOR_EVERY', '100'))
COUNTS = [PIECE_COUNTS[n.split('.')[-1]] for n in MODULES]
MONITOR_BUDGETS = tuple(round(f * sum(COUNTS)) for f in (.001, .003, .01, .03, .1))   # 67M: 43 ... 4301 of 43,008; SS: 7 ... 710 of 7,104
MONITOR_ROWS = slice(1000, 1004)                   # held-out; KLKeep uses val rows 0-127
CHECKPOINT_EVERY = 500
OUT = Path(os.environ.get('MATRYOSHKA_OUT', ROOT/f'output/{EXPERIMENT}/main'))
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
    """Old par-auto concentration on the same scores: mean over matrices of Lp_m / Lp_m(initial pieces). With lp['beta']
    (MATRYOSHKA_FREQ_BETA) each matrix's term is frequency_lp, over its own value for the initial pieces."""
    if lp.get('beta'):
        return sum(frequency_lp(a.float(), lp['p'], lp['epsilon'], lp['beta'], scale, lp['positions']) / d
                   for a, scale, d in zip(parts, lp['scales'], lp['denominators'], strict=True)) / len(parts)
    return sum(method.lp_concentration(a.float(), lp['p'], lp['epsilon']).mean() / d
               for a, d in zip(parts, lp['denominators'], strict=True)) / len(parts)


def frequency_lp(a, p, epsilon, beta, scale, positions):
    """VPD's ImportanceMinimalityLoss shape (its beta term; 0.5 in the published run) on one matrix's scores a [tokens, C]:
        sum_c mean_t a_c^p * (1 + beta * log2(1 + usage_c)),
    usage_c = the piece's sum over `positions` tokens of (a_c / scale)^p, written as its token mean times positions (VPD:
    the batch sum of CI^p). A piece important on many tokens pays more per use. scale (fixed: the matrix's mean |a| on the
    initial pieces) puts usage in units like VPD's CI in [0, 1]. Zero-at-zero powers as lp_concentration; beta = 0 gives
    sum_c mean_t a_c^p, the plain term."""
    plain = ((a.abs() + epsilon).pow(p) - epsilon ** p).mean(0)
    usage = ((a.abs() / scale + epsilon).pow(p) - epsilon ** p).mean(0) * positions
    return (plain * (1 + beta * torch.log2(1 + usage))).sum()


def tail_certificate(cache, pieces, lo, hi, ablated):
    """Signed tail certificate (candidate 1 of the 2026-10-19 robustness write-up) for positions lo:hi of a Fisher cache.
    ablated [n, N]: how far each piece is switched off (1 - mask; 1 on the inactive set of a hard top-k), no gradient.
    Per token and sampled next token j, s_c^j = (r_c.h)(u_c.g_j) is piece c's first-order effect on that sample's loss,
    so a_c^2 = mean_j (s_c^j)^2. Over all 24 matrices at once:
        P_j = sum_c ablated_c max(s_c^j, 0),   N_j = sum_c ablated_c max(-s_c^j, 0),
        cert = (1/2K) sum_j max(P_j, N_j)^2,   quad = (1/2K) sum_j (P_j - N_j)^2.
    quad is the second-order KL of switching the ablated pieces off (on the rank-K Fisher sketch); cert bounds the same
    second-order KL for every mask an adversary can reach by switching ablated pieces back on, fully or partly
    (v in [0, ablated]: sum_c v_c s_c^j lies in [-N_j, P_j]). Returns (cert, quad), each [n] fp32 (fp64 for fp64 inputs); gradients reach the
    pieces' reads and writes. Like the score, g_j is the sampled-label gradient of the whole sequence's loss at this
    position's matrix output, so cross-position terms of the Hessian are left out."""
    total = plus = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for entry, p, v in zip(cache, pieces, ablated.split(COUNTS, -1), strict=True):
            assert entry['kind'] == 'vjp', entry['kind']
            h, cot = entry['h'][lo:hi], entry['events']['cot'][lo:hi]
            if not torch.is_autocast_enabled(p.read.device.type): h, cot = h.to(p.read.dtype), cot.to(p.write.dtype)
            s = ((h @ p.read) * v.to(p.read.dtype))[:, None, :] * (cot @ p.write)     # [n, K, C]
            wide = torch.promote_types(s.dtype, torch.float32)                    # sums in fp32 (fp64 stays fp64)
            t, q = s.sum(-1, dtype=wide), s.clamp_min(0).sum(-1, dtype=wide)
            total, plus = (t, q) if total is None else (total + t, plus + q)
    minus = plus - total
    return torch.maximum(plus, minus).square().mean(-1) / 2, total.square().mean(-1) / 2


def piece_mass(pieces, weight_norms):
    """Per matrix, the sum over its pieces of ||u_c||^2 ||r_c||^2 over ||W||_F^2 (weight_mass67.py's share): 1 when the
    pieces add up to the weights without cancelling, above 1 when large pieces cancel each other. [24]"""
    return torch.stack([(p.read.float().pow(2).sum(0) * p.write.float().pow(2).sum(0)).sum() / n for p, n in zip(pieces, weight_norms)])


def mass_penalty(pieces, weight_norms):
    """Mean over the 24 matrices of (piece mass - 1)^2: pulls every matrix's pieces toward adding up to its weights'
    mass. Both directions: below 1 means pieces split into overlapping copies."""
    return (piece_mass(pieces, weight_norms) - 1).pow(2).mean()


def hard_frequency(mask, positions):
    """VPD's frequency term counted on the hard keep sets: sum_c f_c log2(1 + positions * f_c), f_c = the share of
    these rows (one chunk's positions) whose keep set holds piece c. mask [n, N] is the forward's hard mask, so the
    gradient goes straight through to the scores (the sigmoid top-k Jacobian), as for the masked forward."""
    f = mask.to(torch.promote_types(mask.dtype, torch.float32)).mean(0)
    return (f * torch.log2(1 + positions * f)).sum()


def tail_share(parts, mask):
    """Per token, the share of its total score (all 24 matrices) outside its keep set: sum_c (1 - z_c) a_c / sum_c a_c,
    z the hard mask with no gradient through it; the gradient reaches the scores. [n]"""
    a = torch.cat(parts, -1).float()
    return ((1 - mask.detach()) * a).sum(-1) / a.sum(-1).clamp_min(1e-30)


def routing_masks(generator, shape):
    """VPD's UniformKSubsetRouter (param_decomp.routing.sample_uniform_k_subset_routing_masks): per position k ~ U{1..24},
    then a random k of the 24 matrices are masked. Returns [24, *shape] bool, MODULES order; True = masked."""
    k = torch.randint(1, len(MODULES) + 1, shape, device=generator.device, generator=generator)
    ranks = torch.rand((len(MODULES), *shape), device=generator.device, generator=generator).argsort(0).argsort(0)
    return ranks < k


def masked_logits(model, ids, pieces, alphas, route=None):
    """Same replacement as evaluate_checkpoint.py's KLKeep hooks, with fractional masks allowed. route [24, *ids.shape]
    bool (routing_masks): where False, the matrix keeps its own output (its full weight on the masked model's input)."""
    hooks = []
    try:
        for j, (name, p, a) in enumerate(zip(MODULES, pieces, alphas)):
            def replace(layer, inputs, output, p=p, a=a, r=None if route is None else route[j]):
                masked = ((inputs[0] @ p.read) * a) @ p.write.T
                return masked if r is None else torch.where(r[..., None], masked, output)
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

    def kl(self, model, ids, target, pieces, keep, rows, positions, groups, route=None):
        """The warmup ascent steps on the sources of context slots `rows`, then the KL per budget group at the updated
        sources, with gradients to the pieces, the keep set and the returned sources leaf (for finish). Also returns
        the KL per group at the sources as they came in. positions: the whole batch's, as VPD's mean."""
        gate = keep.detach()
        frozen = [core.FreePieces(p.read.detach(), p.write.detach()) for p in pieces]
        start = None
        for j in range(self.warmup):
            source = self.source[rows].detach().requires_grad_()
            kl = kl_per_position(target, masked_logits(model, ids, frozen, self.masks(gate, source, ids), route))
            if start is None: start = kl.detach().view(groups, -1).mean(-1)
            grad, = torch.autograd.grad(kl.sum() / positions, source)
            self.ascend(rows, grad, self.steps + j + 1)
            del kl, grad
        source = self.source[rows].detach().requires_grad_()
        kl = kl_per_position(target, masked_logits(model, ids, pieces, self.masks(keep, source, ids), route)).view(groups, -1).mean(-1)
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


def adversary_config():
    """vpd_adversary_config with the MATRYOSHKA_ADV_LR / _BETA1 / _BETA2 / _LR_WARMUP overrides on its optimizer."""
    cfg = vpd_adversary_config()
    optimizer = cfg['optimizer']
    for key, value in ADV_OVERRIDES.items():
        if key == 'lr': optimizer['lr_schedule']['start_val'] = value
        elif key == 'warmup_pct': optimizer['lr_schedule']['warmup_pct'] = value
        else: optimizer[key] = value
    return cfg


class MergedAdversary(Adversary):
    """The attacker of the merged loss (main.md section 5). Each update, each context slot is attacked with probability
    `fraction`; an attacked context draws one of its slot's `particles` persistent sources [CONTEXT, N] at random (VPD's
    particle pools, param-decomp commit 177d0cc; 1 particle = one source per slot and position), takes n_warmup_steps
    ascent steps with the pieces frozen, is trained at the updated sources (its only KL), and the same backward gives
    the last ascent step; then the source goes back to its pool. Plain contexts train at the keep mask and leave
    their sources alone. Adam state and the bias-correction count are kept per particle and slot (a slot skips the
    updates where it is plain); otherwise the ascent step is Adversary.ascend."""

    def __init__(self, cfg, fraction, particles, contexts, seed):
        optimizer = cfg['optimizer']
        self.coeff, self.fraction, self.particles = 1., fraction, particles
        self.warmup, self.schedule = cfg['n_warmup_steps'], optimizer['lr_schedule']
        self.beta1, self.beta2, self.eps = optimizer['beta1'], optimizer['beta2'], optimizer['eps']
        generator = torch.Generator(device='cuda').manual_seed(seed)
        self.source = torch.rand(particles, contexts, CONTEXT, sum(COUNTS), device='cuda', generator=generator)
        self.m, self.v = torch.zeros_like(self.source), torch.zeros_like(self.source)
        self.count = torch.zeros(particles, contexts, dtype=torch.long)    # ascent steps so far, per particle and slot
        self.seed, self.lr, self.gradients = seed, 0., None
        self.attacked = torch.zeros(contexts, dtype=torch.bool)
        self.pick = torch.zeros(contexts, dtype=torch.long)

    def draw(self, step):
        """This update's coin per slot and particle per attacked slot, from a generator seeded by the update number."""
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        self.attacked = torch.rand(self.attacked.shape, generator=generator) < self.fraction
        self.pick = torch.randint(self.particles, self.pick.shape, generator=generator)

    def ascend(self, slots, grad):
        """One Adam ascent step on the drawn particles of context slots `slots` (a CPU long tensor), then clamp."""
        if self.gradients is not None: self.gradients.append((slots, grad.detach().clone()))
        index = (self.pick[slots].cuda(), slots.cuda())
        number = (self.count[self.pick[slots], slots] + 1).to('cuda', torch.float64)[:, None, None]
        with torch.no_grad():
            m, v, source = self.m[index], self.v[index], self.source[index]
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            m_hat = m / (1 - self.beta1 ** number).float()
            v_hat = v / (1 - self.beta2 ** number).float()
            source.add_(self.lr * m_hat / v_hat.sqrt().add_(self.eps))
            source.clamp_(0., 1.)
            self.m[index], self.v[index], self.source[index] = m, v, source
        self.count[self.pick[slots], slots] += 1

    def sources(self, slots):
        return self.source[self.pick[slots].cuda(), slots.cuda()]

    def state(self):
        return dict(source=self.source, m=self.m, v=self.v, count=self.count)

    def load(self, state):
        with torch.no_grad():
            for name in ('source', 'm', 'v'): getattr(self, name).copy_(state[name])
        self.count.copy_(state['count'])


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


def accumulate(model, rows, cache, feasible, draws, forward=FORWARD, lp=None, target_logp=None, chunk=None, adversary=None, cert=None,
               merged=None, router=None, freq=None, tail=None):
    """Masked-forward KL over the update's budget groups (len(rows) // len(draws) consecutive contexts share each
    budget draw), run `chunk` contexts at a time; gradients land on the feasible factors' leaves. The loss is the same
    as one group at a time: the sum over groups of (mean KL + lp coeff * lp_term) / their number.
    lp (dict coeff, p, epsilon, denominators) adds coeff * lp_term per group (the chunk's term is the mean of its
    groups' terms). target_logp: intact log-probs of all rows [len(rows)*CONTEXT, V], or None to recompute them.
    adversary (an Adversary for these rows' context slots) adds coeff * its KL per group, at the group's keep set,
    and makes the update's ascent steps on its sources. cert (dict gamma, tau; Fisher cache) adds
    gamma * mean over positions of (tail_certificate - tau)_+ per group, on the pieces the group's mask switches off.
    merged (a MergedAdversary for these rows' slots, drawn for this update; one context per group): its attacked
    contexts take the warm-up ascent steps and are then trained at the attack mask in place of the keep mask, and the
    same backward gives their last ascent step. router (a cuda generator; MATRYOSHKA_ROUTING) draws one routing per
    chunk, used by every pass over that chunk. freq (dict coeff, positions, scale) adds coeff * hard_frequency / scale
    per chunk, weighted like the Lp term (scale None: set to this first chunk's value). tail (dict coeff) adds
    coeff * the mean tail_share per group.

    Returns the leaves (with .grad) and per-group stats."""
    chunk = chunk or CHUNK
    per = len(rows) // len(draws)
    assert per * len(draws) == len(rows) and chunk % per == 0, (len(rows), len(draws), chunk)
    assert adversary is None or adversary.source.shape[0] == len(rows), 'one adversary source slot per context'
    assert merged is None or (merged.source.shape[1] == len(rows) and per == 1 and adversary is None)
    leaves = [core.FreePieces(f.read.detach().requires_grad_(), f.write.detach().requires_grad_()) for f in feasible]
    count = len(draws)
    positions = len(rows) * CONTEXT
    pending = []
    for first in range(0, len(rows), chunk):
        block = rows[first:first+chunk]
        ids = block[:, :-1]
        lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
        ks = draws[first//per:(first+len(block))//per]
        target = intact_logp(model, ids) if target_logp is None else target_logp[lo:hi].view(*ids.shape, -1)
        parts = [] if (lp or tail) else None
        s = log_scores(cache, leaves, lo, hi, parts_out=parts)
        mask, soft = forward_mask(s, ks, forward, per*CONTEXT)
        route = None if router is None else routing_masks(router, tuple(ids.shape))
        extra, train_mask, local = {}, mask, ()
        if merged is not None:
            here = torch.arange(first, first + len(block))
            extra['attacked'] = merged.attacked[here]
            local = extra['attacked'].nonzero().flatten()
            if len(local):
                slots, inside = here[local], local.cuda()
                sub_ids, sub_target = ids[inside], target[inside]
                sub_route = None if route is None else route[:, inside]
                gate = mask.detach().view(len(block), CONTEXT, -1)[inside].flatten(0, 1)
                frozen = [core.FreePieces(p.read.detach(), p.write.detach()) for p in leaves]
                for j in range(merged.warmup):
                    source = merged.sources(slots).requires_grad_()
                    kl = kl_per_position(sub_target, masked_logits(model, sub_ids, frozen, merged.masks(gate, source, sub_ids), sub_route))
                    if j == 0: extra['adv_start'] = kl.detach().view(len(local), -1).mean(-1)
                    grad, = torch.autograd.grad(kl.sum() / positions, source)
                    merged.ascend(slots, grad)
                    del kl, grad, source
                del sub_ids, sub_target, sub_route, gate, frozen
                source = merged.sources(slots).requires_grad_()
                spread = torch.zeros(len(block), CONTEXT, source.shape[-1], device='cuda').index_copy(0, inside, source)
                train_mask = mask + (1 - mask) * spread.flatten(0, 1)      # plain contexts: spread = 0, the keep mask
                del spread
        alphas = [a.reshape(*ids.shape, -1) for a in train_mask.split(COUNTS, -1)]
        kl = kl_per_position(target, masked_logits(model, ids, leaves, alphas, route)).view(len(ks), -1).mean(-1)   # per budget group
        term = lp_term(parts, lp) if lp else None
        loss = kl.sum() if term is None else kl.sum() + lp['coeff'] * term * len(ks)
        if freq is not None:
            raw = hard_frequency(mask, freq['positions'])
            if freq['scale'] is None: freq['scale'] = float(raw.detach())
            loss = loss + freq['coeff'] * (raw / freq['scale']) * len(ks)
            extra['freq'] = raw.detach() / freq['scale']
            del raw
        if tail is not None:
            share = tail_share(parts, mask).view(len(ks), -1).mean(-1)
            loss = loss + tail['coeff'] * share.sum()
            extra['tail'] = share.detach()
            del share
        certs = None
        if cert is not None:
            bound, quad = tail_certificate(cache, leaves, lo, hi, 1 - mask.detach())
            hinge = (bound - cert['tau']).clamp_min(0).view(len(ks), -1).mean(-1)
            loss = loss + cert['gamma'] * hinge.sum()
            certs = (bound.detach().view(len(ks), -1).mean(-1), quad.detach().view(len(ks), -1).mean(-1), hinge.detach())
            del bound, quad, hinge
        adv = None
        if adversary is not None:
            here = slice(first, first + len(block))
            adv_kl, adv_start, source = adversary.kl(model, ids, target, leaves, soft if ADV_GATE == 'soft' else mask, here, positions, len(ks), route)
            loss = loss + adversary.coeff * adv_kl.sum()
            adv = (adv_kl.detach(), adv_start)
        (loss / count).backward()
        if adversary is not None:
            adversary.finish(here, source)
            del adv_kl, adv_start, source
        if len(local):
            merged.ascend(slots, source.grad)      # (KL sum / positions)'s gradient, as the warm-up steps
            del source
        with torch.no_grad():
            on = soft > .5
            pending.append((ks, kl.detach(), on.sum(-1).float().view(len(ks), -1).mean(-1),
                            (soft * ~on).sum(-1).view(len(ks), -1).mean(-1), None if term is None else term.detach(), adv, certs,
                            extra, local))
        del target, s, mask, soft, alphas, kl, on, parts, term, loss, adv, certs, train_mask, route
    if adversary is not None: adversary.steps += adversary.warmup + 1
    stats = []                                       # one host sync for the whole update
    for ks, kl, on, leak, term, adv, certs, extra, local in pending:
        kl, on, leak = kl.tolist(), on.tolist(), leak.tolist()
        term = None if term is None else float(term)
        adv = None if adv is None else [a.tolist() for a in adv]
        certs = None if certs is None else [c.tolist() for c in certs]
        starts = dict(zip(local.tolist(), extra['adv_start'].tolist())) if 'adv_start' in extra else {}
        for j, k in enumerate(ks):
            stats.append(dict(k=k, kl=kl[j], on_over_k=on[j] / k, soft_leak=leak[j] / k))
            if term is not None: stats[-1]['lp_normalized'] = term
            if adv is not None: stats[-1].update(adv_kl=adv[0][j], adv_kl_start=adv[1][j])
            if certs is not None: stats[-1].update(cert=certs[0][j], cert_quad=certs[1][j], cert_hinge=certs[2][j])
            if 'freq' in extra: stats[-1]['freq_normalized'] = float(extra['freq'])
            if 'tail' in extra: stats[-1]['tail_share'] = float(extra['tail'][j])
            if 'attacked' in extra: stats[-1]['attacked'] = bool(extra['attacked'][j])
            if j in starts: stats[-1]['adv_kl_start'] = starts[j]
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
    if TARGET == 'pile67':
        control = json.loads((ROOT/'output/gim_1h/plan.json').read_text())
        assert control['sensor'] == 'gim' and control['rules'] == RULES
        assert control['target_sha256'] == TARGET_SHA and control['data_sha256'] == manifest['train']['sha256']
        assert control['component_counts'] == dict(zip(MODULES, COUNTS))
        cfg = method.TrainConfig(**control['training_config'])
    else:
        from common import TRAINING_CONFIG
        cfg = method.TrainConfig(**TRAINING_CONFIG)
    assert cfg.steps == 8700 and cfg.precision == 'bf16' and cfg.projection_precision == 'fp32'
    assert BATCH % BUDGET_CONTEXTS == 0
    cfg = replace(cfg, steps=SCHEDULE_STEPS, projection_precision=PROJECTION)    # defaults: the control's own
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
    if FREQ_BETA:
        plan['lp']['frequency'] = dict(
            beta=FREQ_BETA, form='per matrix: sum_c mean_t a_c^p * (1 + beta * log2(1 + usage_c)), usage_c = sum over the '
                                 f'update\'s {BATCH*CONTEXT} positions of (a_c / scale)^p (token mean of the chunk times that count)',
            scale='each matrix\'s mean |a| on the initial pieces at the reference set (fixed)', vpd='ImportanceMinimalityLoss beta term',
            normalization='each matrix over its own frequency-weighted value for the initial pieces on the reference set at the '
                          'current p, so the total pressure starts like the plain term and the beta factor shifts it to frequent pieces')
    if MASS_COEFF:
        plan['mass'] = dict(coeff=MASS_COEFF, term='mean over the 24 matrices of (sum_c ||u_c||^2 ||r_c||^2 / ||W||_F^2 - 1)^2, '
                                                   'on the projected pieces, once per update, gradient through the projection')
    if CERT_GAMMA:
        plan['certificate'] = dict(
            gamma=CERT_GAMMA, tau=CERT_TAU,
            term='gamma * mean over positions of (cert - tau)_+, per budget group; cert = (1/2K) sum_j max(P_j, N_j)^2 over the '
                 'K sampled next tokens, P_j / N_j = sums of the positive / negative parts of s_c^j = (r_c.h)(u_c.g_j) over the '
                 'pieces the group\'s mask switches off (weighted by 1 - mask, no gradient through the mask), all 24 matrices together',
            meaning='second-order bound, on the rank-K Fisher sketch at the intact model, on the KL of every mask that switches '
                    'switched-off pieces back on (fully or partly); write-up candidate 1, real-model version (no variant 1b)')
    if SCHEDULE_STEPS != 8700 or PROJECTION != 'fp32':
        plan['schedule'] = dict(steps=SCHEDULE_STEPS, projection=PROJECTION,
                                note='the lr cosine (1e-3 -> 1e-5), the p schedule (1 -> 0.5 over its first 80%) and the attacker\'s lr '
                                     'warm-up run over this horizon instead of the control\'s 8,700 steps')
    if ADV_FRACTION:
        merged_cfg = adversary_config()
        plan['merged_adversary'] = dict(
            fraction=ADV_FRACTION, particles=PARTICLES, optimizer=merged_cfg['optimizer'], n_warmup_steps=merged_cfg['n_warmup_steps'],
            loss='mean over all contexts of KL(p_intact || p_masked): attacked contexts (each with probability fraction, per '
                 'update) at keep + (1 - keep) * source on the routed matrices, plain ones at the keep mask; no other KL term',
            sources=f'[{PARTICLES}, {BATCH}, {CONTEXT}, {total}] fp32, uniform [0, 1] (cuda generator seed {cfg.seed + 606}), kept '
                    'across updates; an attacked context uses one particle of its slot, drawn at random, and writes it back',
            update=f'{merged_cfg["n_warmup_steps"]} ascent steps with the pieces frozen, then the training forward at the updated '
                   'sources, whose backward also gives one more ascent step; Adam on the batch-mean KL, clamped to [0, 1], '
                   'state and bias-correction count per particle and slot',
            source_note='VPD PersistentPGDReconLoss (persistent sources, 2 + 1 steps, mask form); particle pools from VPD '
                        'param-decomp commit 177d0cc', lr_schedule_horizon=cfg.steps)
    if ROUTING != 'all':
        plan['routing'] = dict(kind=ROUTING, rule='per position k ~ U{1..24}, a random k of the 24 matrices masked; the others '
                                                  'output W on the masked model\'s input (VPD UniformKSubsetRouter, App. A.3.3)',
                               scope='one draw per chunk and update, shared by the plain and attacked forwards and the warm-up steps')
    if FREQ_COEFF:
        plan['hard_frequency'] = dict(coeff=FREQ_COEFF, positions=BATCH*CONTEXT,
                                      term='per chunk, sum_c f_c log2(1 + positions * f_c) / F0, f_c = share of the chunk\'s positions '
                                           'whose hard keep set holds c (straight-through gradient); F0 = its value on the first chunk of '
                                           'the first update; weighted per chunk like the Lp term')
    if TAIL_COEFF:
        plan['tail_share'] = dict(coeff=TAIL_COEFF, term='mean over positions of sum_c (1 - z_c) a_c / sum_c a_c (all 24 matrices, '
                                                         'z the hard keep set, no gradient through z), per budget group')
    if INIT:
        plan['init'] = dict(checkpoint=INIT, note='pieces, Adam state, budget generator, RNG states and step count (and the '
                                                  'adversary\'s sources when both runs have one) of that checkpoint; the lr '
                                                  'schedule continues from its step')
    if TARGET != 'pile67':
        # The plan's wording is the 67M run's; restate its counts, and drop the 67M controls this target has no copy of.
        text = json.dumps(plan)
        for old, new in (('24 matrices', f'{len(MODULES)} matrices'), ('U{1..24}', f'U{{1..{len(MODULES)}}}'), ('43,008', f'{total:,}')):
            text = text.replace(old, new)
        plan = dict(json.loads(text), target=TARGET, matched_control=None, initialization='seed-0 projected factors (core.initialize_pieces)',
                    data='train rows in order, one row per context, never repeated')
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
    merged = MergedAdversary(merged_cfg, ADV_FRACTION, PARTICLES, BATCH, seed=cfg.seed + 606) if ADV_FRACTION else None
    freq = dict(coeff=FREQ_COEFF, positions=BATCH*CONTEXT, scale=None) if FREQ_COEFF else None
    scales = [s.abs().mean().clamp_min(1e-30) for s in initial_scores] if FREQ_BETA else None
    start = 0
    if resuming or INIT:
        if resuming:
            saved_plan = json.loads((OUT/'plan.json').read_text())
            assert {k: v for k, v in saved_plan.items() if k not in ('gpu', 'job_ids')} == json.loads(json.dumps({k: v for k, v in plan.items() if k != 'gpu'}))
        source = json.loads((OUT/'latest_checkpoint.json').read_text())['full_checkpoint'] if resuming else INIT
        full = torch.load(source, map_location='cpu', weights_only=False)
        assert full['data_sha256'] == manifest['train']['sha256'] and full['target_sha256'] == TARGET_SHA and full['sensor'] == SENSOR
        with torch.no_grad():
            for p, s in zip(pieces, full['state']['pieces'], strict=True):
                p.read.copy_(s['read']); p.write.copy_(s['write'])
        optimizer.load_state_dict(full['state']['optimizer'])
        budgets.set_state(full['state']['budgets'])
        if adversary is not None and 'adversary' in full['state']: adversary.load(full['state']['adversary'])
        else: assert adversary is None or not resuming, 'resuming an adversarial run needs its sources'
        if merged is not None and 'merged_adversary' in full['state']: merged.load(full['state']['merged_adversary'])
        else: assert merged is None or not resuming, 'resuming a merged-loss run needs its sources'
        if freq is not None: freq['scale'] = full['state'].get('freq_scale')
        assert freq is None or freq['scale'] is not None or not resuming, 'resuming needs the frequency scale F0'
        torch.set_rng_state(full['torch_rng']); torch.cuda.set_rng_state(full['cuda_rng'])
        start = full['completed']
        del full
        print(json.dumps(dict(resumed_from=start) if resuming else dict(initialized_from=INIT, step=start)), flush=True)
        assert start < TARGET_STEPS, (start, TARGET_STEPS)
    if not resuming:
        save_json(plan, OUT/'plan.json'); print(json.dumps(dict(plan=plan)), flush=True)
    previous = json.loads((OUT/'wandb.json').read_text()) if resuming else None
    name = f'parauto-matryoshka-{FORWARD}-fisher-k4-5000' if OUT.name == 'main' else f'parauto-matryoshka-{OUT.name}'
    run = wandb.init(project='matryoshka', entity='rohan-kathuria-neu', group=EXPERIMENT,
                     name=name if TARGET == 'pile67' else f'ss-{OUT.name}', job_type='training',
                     dir=os.environ.get('MATRYOSHKA_WANDB_DIR', str(OUT)), config=plan,
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
        if merged is not None: state['merged_adversary'] = merged.state()
        if freq is not None: state['freq_scale'] = freq['scale']
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
        if merged is not None: merged.set_lr(step, cfg.steps); merged.draw(step)
        router = torch.Generator(device='cuda').manual_seed(cfg.seed + 707 + 1_000_003 * step) if ROUTING != 'all' else None
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
            if FREQ_BETA:
                lp.update(beta=FREQ_BETA, scales=scales, positions=BATCH*CONTEXT,
                          denominators=[frequency_lp(s, p_now, cfg.lp_epsilon, FREQ_BETA, scale, BATCH*CONTEXT).clamp_min(1e-8)
                                        for s, scale in zip(initial_scores, scales)])
        leaves, stats = accumulate(model, rows, cache, feasible, draws, lp=lp, target_logp=target_logp, adversary=adversary,
                                   cert=dict(gamma=CERT_GAMMA, tau=CERT_TAU) if CERT_GAMMA else None, merged=merged, router=router,
                                   freq=freq, tail=dict(coeff=TAIL_COEFF) if TAIL_COEFF else None)
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
        if merged is not None:
            hit = [s for s in stats if s['attacked']]
            plain = [s for s in stats if not s['attacked']]
            row.update(attacked_share=len(hit) / len(stats), adv_lr=merged.lr, adv_source_mean=float(merged.source.mean()))
            if hit: row.update(kl_attacked=float(np.mean([s['kl'] for s in hit])), kl_attacked_start=float(np.mean([s['adv_kl_start'] for s in hit])))
            if plain: row['kl_plain'] = float(np.mean([s['kl'] for s in plain]))
        if freq is not None: row.update(freq_normalized=float(np.mean([s['freq_normalized'] for s in stats])), freq_scale=freq['scale'])
        if TAIL_COEFF: row['tail_share'] = float(np.mean([s['tail_share'] for s in stats]))
        if CERT_GAMMA:
            # cancellation index: cert over the second-order KL of the top-k mask itself (1 = no cancellation in the tail)
            row.update(cert=float(np.mean([s['cert'] for s in stats])), cert_quad=float(np.mean([s['cert_quad'] for s in stats])),
                       cert_hinge=float(np.mean([s['cert_hinge'] for s in stats])))
            row['cert_index'] = row['cert'] / max(row['cert_quad'], 1e-30)
        for lo, hi, label in ((0, 100, 'lt100'), (100, 1000, '100_1000'), (1000, 10000, '1000_10000'), (10000, total, 'ge10000')):
            inside = [s for s in stats if lo <= s['k'] < hi]
            if inside: row[f'kl_k_{label}'] = float(np.mean([s['kl'] for s in inside]))
            if inside and adversary is not None: row[f'adv_kl_k_{label}'] = float(np.mean([s['adv_kl'] for s in inside]))
            if merged is not None:
                for kind, flag in (('attacked', True), ('plain', False)):
                    picked = [s['kl'] for s in inside if s['attacked'] == flag]
                    if picked: row[f'kl_{kind}_k_{label}'] = float(np.mean(picked))
            if inside and CERT_GAMMA:
                row[f'cert_k_{label}'] = float(np.mean([s['cert'] for s in inside]))
                row[f'cert_quad_k_{label}'] = float(np.mean([s['cert_quad'] for s in inside]))
        if not all(math.isfinite(v) for v in row.values()): raise RuntimeError(f'Nonfinite metrics: {row}')
        run.log({'update': completed, **{'train/'+k: v for k, v in row.items()}})
        if completed % 10 == 0 or completed <= 5:
            extra = {} if adversary is None else dict(adv_kl_by_budget=[s['adv_kl'] for s in stats])
            if merged is not None: extra['attacked_by_budget'] = [int(s['attacked']) for s in stats]
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
