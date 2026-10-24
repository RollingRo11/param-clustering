"""Checks for matryoshka67.py: the sigmoid top-k threshold and its implicit gradient, the all-kept
masked forward, gradient flow through one micro-batch into every matrix's free factors, the sped-up
update (Triton bisection, topk threshold, chunked micro-batches, cached KL target) against the
one-micro-batch-at-a-time implementation it replaced (reference_* below, as of commit 3c2ccc6), the
budget sampler (power, stratification, one budget per context), the persistent PGD loss against VPD's own code, and the
signed tail certificate (against loops and a brute-force worst case, then inside one real update)."""
import json
import math

import numpy as np
import torch

import matryoshka67 as mt
from common import MODULES, CONTEXT, target, tokens, batch
from fisher_attribution import capture_fisher
from train_pair import core, method

# Four budgets from 0.1% to 70% of the pieces (40 ... 30,000 of the 67M target's 43,008), scaled to the target in use.
DRAWS = [max(1, round(k * sum(mt.COUNTS) / 43008)) for k in (40, 430, 4301, 30000)]


def reference_sigmoid_topk(s, k, iterations=40):
    with torch.no_grad():
        lo = -s.max(-1, keepdim=True).values - 50.
        hi = -s.min(-1, keepdim=True).values + 50.
        for _ in range(iterations):
            mid = (lo + hi) / 2
            over = torch.sigmoid(s + mid).sum(-1, keepdim=True) > k
            hi, lo = torch.where(over, mid, hi), torch.where(over, lo, mid)
        c0 = (lo + hi) / 2
        slope = torch.sigmoid(s + c0)
        slope = slope * (1 - slope)
    shift = (slope * (s - s.detach())).sum(-1, keepdim=True) / slope.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.sigmoid(s + c0 - shift)


def reference_forward_mask(s, k, forward):
    soft = reference_sigmoid_topk(s, k)
    if forward == 'soft': return soft, soft
    with torch.no_grad():
        hard = (s >= s.sort(-1, descending=True).values[:, k-1:k]).to(s.dtype)
    return hard + soft - soft.detach(), soft


def reference_accumulate(model, rows, cache, feasible, draws, forward, lp):
    leaves = [core.FreePieces(f.read.detach().requires_grad_(), f.write.detach().requires_grad_()) for f in feasible]
    micro = range(0, len(rows), mt.MICRO)
    stats = []
    for j, first in enumerate(micro):
        block = rows[first:first+mt.MICRO]
        ids = block[:, :-1]
        target_logp = mt.intact_logp(model, ids)
        parts = [] if lp else None
        s = mt.log_scores(cache, leaves, first*CONTEXT, (first+len(block))*CONTEXT, parts_out=parts)
        mask, soft = reference_forward_mask(s, draws[j], forward)
        alphas = [a.reshape(*ids.shape, -1) for a in mask.split(mt.COUNTS, -1)]
        kl = mt.kl_per_position(target_logp, mt.masked_logits(model, ids, leaves, alphas)).mean()
        term = mt.lp_term(parts, lp) if lp else None
        ((kl if term is None else kl + lp['coeff'] * term) / len(micro)).backward()
        stats.append(dict(k=draws[j], kl=float(kl), lp_normalized=None if term is None else float(term)))
    return leaves, stats


def check_bisection_kernel():
    """Triton bisection against the eager loop (float32 on the CPU) on rows the size of the real score rows."""
    g = torch.Generator(device='cuda').manual_seed(0)
    s = torch.randn(64, sum(mt.COUNTS), device='cuda', generator=g) * 4
    ks = torch.randint(32, sum(mt.COUNTS) - 1, (64,), device='cuda', generator=g).float()
    fast = mt.threshold_shift(s, ks)
    eager = mt.threshold_shift(s.cpu(), ks.cpu()).cuda()
    a_fast, a_eager = torch.sigmoid(s + fast), torch.sigmoid(s + eager)
    result = dict(alpha_max_abs_diff=float((a_fast - a_eager).abs().max()),
                  row_sum_error_triton=float((a_fast.sum(-1) - ks).abs().max()),
                  row_sum_error_eager=float((a_eager.sum(-1) - ks).abs().max()))
    assert result['alpha_max_abs_diff'] < 1e-4 and result['row_sum_error_triton'] < 0.05, result
    return result


def relative(a, b):
    return max(float((x - y).norm() / y.norm().clamp_min(1e-30)) for x, y in zip(a, b, strict=True))


def check_speedups(model, data, weights):
    """Same loss and gradients as reference_accumulate on 16 real contexts (4 budgets, hard mask, Lp term on).
    chunk=4 isolates the new kernels and the cached KL target; chunk=16 adds bigger bf16 matmuls, whose rounding
    differs from four smaller ones."""
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    cfg = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:16]
    draws = list(DRAWS)
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    # The cache's own forward gives the old separate intact forward's log-probs (same 4-context shapes here).
    separate = torch.cat([mt.intact_logp(model, rows[i:i+mt.MICRO, :-1]) for i in range(0, len(rows), mt.MICRO)])
    result = dict(kl_target_max_abs_diff=float((logp.view(16, CONTEXT, -1) - separate).abs().max()))
    assert result['kl_target_max_abs_diff'] < 1e-3, result
    del separate
    feasible = [method.project(w, p, cfg) for w, p in zip(weights, pieces)]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        own = [method.structured.scores(method.structured.slice_entry(e, slice(0, 256)), f).float() for e, f in zip(cache, feasible)]
    lp = dict(coeff=1., p=0.8, epsilon=1e-6, denominators=[method.lp_concentration(s, 0.8, 1e-6).mean() for s in own])
    runs = dict(reference=reference_accumulate(model, rows, cache, feasible, draws, 'hard', lp),
                chunk4=mt.accumulate(model, rows, cache, feasible, draws, 'hard', lp, target_logp=logp, chunk=4),
                chunk16=mt.accumulate(model, rows, cache, feasible, draws, 'hard', lp, target_logp=logp, chunk=16))
    grads = {k: [t.grad for l in leaves for t in (l.read, l.write)] for k, (leaves, _) in runs.items()}
    kls = {k: torch.tensor([s['kl'] for s in stats]) for k, (_, stats) in runs.items()}
    for label in ('chunk4', 'chunk16'):
        result[label] = dict(kl=kls[label].tolist(), kl_max_rel_diff=float(((kls[label] - kls['reference']).abs() / kls['reference']).max()),
                             grad_max_rel_diff=relative(grads[label], grads['reference']),
                             lp_normalized=runs[label][1][0]['lp_normalized'], lp_reference=runs['reference'][1][0]['lp_normalized'])
    result['reference_kl'] = kls['reference'].tolist()
    assert result['chunk4']['kl_max_rel_diff'] < 1e-3 and result['chunk4']['grad_max_rel_diff'] < 1e-2, result
    assert result['chunk16']['kl_max_rel_diff'] < 1e-2 and result['chunk16']['grad_max_rel_diff'] < 5e-2, result
    # The Fisher cache itself: 16 contexts per forward/backward vs 4, with the true next tokens as labels (no sampling).
    small = capture_fisher(model, rows, k=1, microbatch=4, labels=True)
    large = capture_fisher(model, rows, k=1, microbatch=16, labels=True)
    result['cache_chunk16_vs_4'] = dict(
        h_max_rel_diff=relative([e['h'].float() for e in large], [e['h'].float() for e in small]),
        cot_max_rel_diff=relative([e['events']['cot'].float() for e in large], [e['events']['cot'].float() for e in small]))
    # bf16 rounding of bigger matmuls; the SimpleStories target's activations land at about 1.05e-2 (Oct 2026, RTX PRO 4500).
    h_tolerance = 1e-2 if len(MODULES) == 24 else 2e-2
    assert result['cache_chunk16_vs_4']['h_max_rel_diff'] < h_tolerance and result['cache_chunk16_vs_4']['cot_max_rel_diff'] < 5e-2, result
    # One budget per context: when each group of 4 contexts draws the same budget, it is the same update as one per 4.
    leaves, stats = mt.accumulate(model, rows, cache, feasible, [k for k in draws for _ in range(4)], 'hard', lp, target_logp=logp, chunk=16)
    per_context_kl = torch.tensor([s['kl'] for s in stats]).view(4, 4).mean(-1)
    result['per_context_vs_per4'] = dict(kl_max_rel_diff=float(((per_context_kl - kls['chunk16']).abs() / kls['chunk16']).max()),
                                         grad_max_rel_diff=relative([t.grad for l in leaves for t in (l.read, l.write)], grads['chunk16']))
    assert result['per_context_vs_per4']['kl_max_rel_diff'] < 1e-4 and result['per_context_vs_per4']['grad_max_rel_diff'] < 1e-3, result
    return result


def reference_adversary(model, rows, cache, feasible, draws, lp, target_logp, cfg, coeff, init, lr):
    """VPD's own persistent PGD code on the whole batch at once, in train_pair.VPD.step's order: n_warmup_steps ascent
    steps (param_decomp's AdamPGDOptimizer and _interpolate_component_mask, then clamp), the loss, the sources'
    gradient by autograd.grad, backward, the last ascent step. Returns the leaves, the sources after the update,
    every ascent step's gradient [len(rows), CONTEXT, N] and the adversarial KL per context."""
    from param_decomp.configs import AdamPGDConfig
    from param_decomp.persistent_pgd import AdamPGDOptimizer, _interpolate_component_mask
    leaves = [core.FreePieces(f.read.detach().requires_grad_(), f.write.detach().requires_grad_()) for f in feasible]
    frozen = [core.FreePieces(l.read.detach(), l.write.detach()) for l in leaves]
    ids, per = rows[:, :-1], len(rows) // len(draws)
    target = target_logp.view(*ids.shape, -1)
    optimizer = AdamPGDOptimizer(AdamPGDConfig(**cfg['optimizer']))
    sources = {n: part.clone().requires_grad_() for n, part in zip(MODULES, init.split(mt.COUNTS, -1))}
    optimizer.init_state(sources); optimizer.set_lr(lr)
    parts = []
    s = mt.log_scores(cache, leaves, 0, len(rows)*CONTEXT, parts_out=parts)
    keep, _ = mt.forward_mask(s, draws, 'hard', per*CONTEXT)
    gates = {n: g.reshape(*ids.shape, -1) for n, g in zip(MODULES, keep.split(mt.COUNTS, -1))}

    def kl_at(gates, pieces):
        masks = _interpolate_component_mask(gates, sources)
        return mt.kl_per_position(target, mt.masked_logits(model, ids, pieces, [masks[n] for n in MODULES]))

    seen = []
    def ascend(grads):
        seen.append(torch.cat([grads[n] for n in MODULES], -1))
        with torch.no_grad():
            optimizer.step(sources, grads)
            for x in sources.values(): x.clamp_(0., 1.)

    for _ in range(cfg['n_warmup_steps']):
        loss = kl_at({n: g.detach() for n, g in gates.items()}, frozen).mean()
        ascend(dict(zip(MODULES, torch.autograd.grad(loss, list(sources.values())))))
    adv = kl_at(gates, leaves)
    grads = dict(zip(MODULES, torch.autograd.grad(adv.mean(), list(sources.values()), retain_graph=True)))
    kl = mt.kl_per_position(target, mt.masked_logits(model, ids, leaves, list(gates.values()))).view(len(draws), -1).mean(-1)
    (kl.mean() + lp['coeff'] * mt.lp_term(parts, lp) + coeff * adv.mean()).backward()
    ascend(grads)
    return leaves, torch.cat([sources[n].detach() for n in MODULES], -1), seen, adv.detach().view(len(draws), -1).mean(-1)


def check_adversary(model, data, weights):
    """check_hard_adversary with the hard gate (VPD's comparison is only defined for it), then, when the run uses
    MATRYOSHKA_ADV_GATE=soft, check_soft_gate."""
    gate = mt.ADV_GATE
    try:
        mt.ADV_GATE = 'hard'
        result = check_hard_adversary(model, data, weights)
    finally:
        mt.ADV_GATE = gate
    if gate == 'soft': result['soft_gate'] = check_soft_gate(model, data, weights)
    return result


def check_soft_gate(model, data, weights):
    """MATRYOSHKA_ADV_GATE=soft: with sources at 0 the adversarial forward is the soft (sigmoid top-k alpha) forward,
    while the main forward stays hard; the update's gradient is finite."""
    cfg = mt.vpd_adversary_config()
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:4]
    draws = list(DRAWS)
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
    _, soft_stats = mt.accumulate(model, rows, cache, feasible, draws, 'soft', None, target_logp=logp, chunk=4)
    _, hard_stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=4)
    zero = mt.Adversary(cfg, 1., 4, seed=0); zero.source.zero_(); zero.lr = 0.
    leaves, stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=4, adversary=zero)
    result = dict(adv_vs_soft_kl_max_rel_diff=max(abs(a['adv_kl'] - b['kl']) / b['kl'] for a, b in zip(stats, soft_stats)),
                  main_vs_hard_kl_max_rel_diff=max(abs(a['kl'] - b['kl']) / b['kl'] for a, b in zip(stats, hard_stats)),
                  soft_kl=[s['kl'] for s in soft_stats], hard_kl=[s['kl'] for s in hard_stats])
    assert result['adv_vs_soft_kl_max_rel_diff'] < 1e-6 and result['main_vs_hard_kl_max_rel_diff'] < 1e-6, result
    assert all(torch.isfinite(t.grad).all() for l in leaves for t in (l.read, l.write)), result
    return result


def check_hard_adversary(model, data, weights):
    """The persistent PGD loss against VPD's own code: the lr schedule (get_scheduled_value), the ascent step on slices
    of the sources (AdamPGDOptimizer + clamp, bit for bit), the update on 4 real contexts with per-context budgets
    (chunk 4: same shapes as the reference; chunk 2: two chunks), and sources at 0 giving back the masked forward."""
    from param_decomp.configs import AdamPGDConfig, ScheduleConfig
    from param_decomp.persistent_pgd import AdamPGDOptimizer
    from param_decomp.utils.general_utils import get_scheduled_value
    cfg, result = mt.vpd_adversary_config(), {}
    adversary = mt.Adversary(cfg, 1., 4, seed=0)
    schedule = ScheduleConfig(**cfg['optimizer']['lr_schedule'])
    for step in (0, 1, 100, 216, 217, 218, 4999, 8699):
        adversary.set_lr(step, 8700)
        assert adversary.lr == get_scheduled_value(step, 8700, schedule), (step, adversary.lr)
    result['lr_warmup_steps'] = int(8700 * schedule.warmup_pct)
    # Ascent steps on two halves of the context slots, with shared step numbers, are VPD's full-batch steps.
    g = torch.Generator(device='cuda').manual_seed(1)
    theirs = {'all': adversary.source.clone()}
    optimizer = AdamPGDOptimizer(AdamPGDConfig(**cfg['optimizer'])); optimizer.init_state(theirs); optimizer.set_lr(.01)
    adversary.lr = .01
    for number in (1, 2, 3):          # gradient sizes from far below Adam's eps (1e-8) to far above it
        scale = torch.empty(adversary.source.shape, device='cuda').uniform_(-11, -4, generator=g)
        grad = torch.randn(adversary.source.shape, device='cuda', generator=g) * torch.pow(10., scale)
        optimizer.step(theirs, {'all': grad}); theirs['all'].clamp_(0., 1.)
        for rows in (slice(0, 2), slice(2, 4)): adversary.ascend(rows, grad[rows], number)
    assert torch.equal(adversary.source, theirs['all']), float((adversary.source - theirs['all']).abs().max())
    del theirs, optimizer, grad, scale
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:4]
    draws = list(DRAWS)
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        own = [method.structured.scores(method.structured.slice_entry(e, slice(0, 256)), f).float() for e, f in zip(cache, feasible)]
    lp = dict(coeff=1., p=0.8, epsilon=1e-6, denominators=[method.lp_concentration(s, 0.8, 1e-6).mean() for s in own])
    # Sources at 0 and no ascent: the adversarial forward is the masked forward, so the gradient doubles.
    plain, plain_stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=4)
    zero = mt.Adversary(cfg, 1., 4, seed=0); zero.source.zero_(); zero.lr = 0.
    doubled, doubled_stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=4, adversary=zero)
    result['zero_sources'] = dict(
        kl_max_rel_diff=max(abs(a['adv_kl'] - b['kl']) / b['kl'] for a, b in zip(doubled_stats, plain_stats)),
        grad_max_rel_diff=relative([t.grad for l in doubled for t in (l.read, l.write)], [2 * t.grad for l in plain for t in (l.read, l.write)]))
    assert result['zero_sources']['kl_max_rel_diff'] < 1e-6 and result['zero_sources']['grad_max_rel_diff'] < 1e-5, result
    assert zero.steps == cfg['n_warmup_steps'] + 1 and float(zero.source.abs().max()) == 0.
    del plain, doubled, zero
    # One update against VPD's code, from the same uniform sources.
    init = torch.rand(4, CONTEXT, sum(mt.COUNTS), device='cuda', generator=torch.Generator(device='cuda').manual_seed(2))

    def ours(coeff, lr, chunk):
        adversary = mt.Adversary(cfg, coeff, 4, seed=0); adversary.source.copy_(init); adversary.lr = lr; adversary.gradients = []
        leaves, stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', lp, target_logp=logp, chunk=chunk, adversary=adversary)
        numbers = sorted({n for _, n, _ in adversary.gradients})
        assert numbers == list(range(1, cfg['n_warmup_steps'] + 2)) and adversary.steps == numbers[-1], numbers
        grads = [torch.cat([grad for _, n, grad in sorted(adversary.gradients, key=lambda x: x[0].start) if n == number]) for number in numbers]
        return leaves, stats, grads, adversary.source

    def compare(label, mine, theirs):
        (leaves, stats, grads, source), (ref_leaves, ref_source, ref_grads, ref_kl) = mine, theirs
        kl = torch.tensor([s['adv_kl'] for s in stats])
        moved = (source - ref_source).abs()
        result[label] = dict(
            adv_kl=kl.tolist(), adv_kl_start=[s['adv_kl_start'] for s in stats], adv_kl_reference=ref_kl.tolist(),
            adv_kl_max_rel_diff=float(((kl - ref_kl.cpu()).abs() / ref_kl.cpu()).max()),
            source_grad_rel_diff_per_step=[relative([a], [b]) for a, b in zip(grads, ref_grads, strict=True)],
            piece_grad_max_rel_diff=relative([t.grad for l in leaves for t in (l.read, l.write)], [t.grad for l in ref_leaves for t in (l.read, l.write)]),
            source_max_abs_diff=float(moved.max()), source_fraction_off_by_1e_4=float((moved > 1e-4).float().mean()),
            source_mean_move=float((source - init).abs().mean()))
        return result[label]

    # Weight 1 (this run's): the adversarial KL's backward has VPD's scale exactly, so one chunk of 4 contexts is VPD bit for bit.
    reference = reference_adversary(model, rows, cache, feasible, draws, lp, logp, cfg, 1., init, .01)
    exact = compare('weight1_chunk4', ours(1., .01, 4), reference)
    assert exact['adv_kl_max_rel_diff'] < 1e-6 and max(exact['source_grad_rel_diff_per_step']) < 1e-6, result
    assert exact['piece_grad_max_rel_diff'] < 1e-4 and exact['source_max_abs_diff'] < 1e-6, result
    # Two chunks, reported only: bf16 matmul shapes change the gradients by ~1%, and the sign-like first ascent steps turn
    # that into different sources wherever a gradient is near zero (checked below with the sources held still).
    compare('weight1_chunk2', ours(1., .01, 2), reference)
    del reference
    # Weight 0.7: the pieces' gradient scales exactly; the sources' gradient (taken from the same backward, divided by the
    # weight) matches VPD's separate unscaled backward only up to bf16 rounding (~1% at the last step, the only scaled one).
    scaled = compare('weight0.7_chunk4', ours(.7, .01, 4), reference_adversary(model, rows, cache, feasible, draws, lp, logp, cfg, .7, init, .01))
    assert scaled['adv_kl_max_rel_diff'] < 1e-6 and scaled['piece_grad_max_rel_diff'] < 1e-4, result
    assert max(scaled['source_grad_rel_diff_per_step'][:-1]) < 1e-6 and scaled['source_grad_rel_diff_per_step'][-1] < 3e-2, result
    # Chunking with the sources held still (lr 0): two chunks of 2 against VPD's whole batch, up to bf16 rounding.
    # A wrong slice or normalization would put the sources' gradient off by a factor (>= 50%).
    still = compare('weight1_chunk2_lr0', ours(1., 0., 2), reference_adversary(model, rows, cache, feasible, draws, lp, logp, cfg, 1., init, 0.))
    assert still['adv_kl_max_rel_diff'] < 1e-2 and max(still['source_grad_rel_diff_per_step']) < 5e-2, result
    assert still['piece_grad_max_rel_diff'] < 5e-2 and still['source_max_abs_diff'] == 0., result
    return result


def check_frequency():
    """frequency_lp: beta 0 is the plain Lp term; beta > 0 is VPD's sum_c mean_c (1 + beta log2(1 + usage_c)) term by term
    (float64 loop); lp_term with a beta averages frequency_lp / denominator over matrices; the gradient is finite."""
    g = torch.Generator().manual_seed(11)
    parts = [torch.rand(256, n, generator=g, dtype=torch.float64).pow(6) * 3 for n in (40, 70)]   # heavy-tailed scores
    p, eps, beta, positions = 0.7, 1e-6, 0.5, 32768
    for a in parts:
        plain = mt.frequency_lp(a, p, eps, 0., 1., positions)
        assert torch.allclose(plain, method.lp_concentration(a, p, eps).mean(), rtol=1e-10), plain
        scale = a.abs().mean()
        expected = 0.
        for c in range(a.shape[1]):
            mean_c = float(((a[:, c] + eps) ** p - eps ** p).mean())
            usage_c = float(((a[:, c] / scale + eps) ** p - eps ** p).mean()) * positions
            expected += mean_c * (1 + beta * math.log2(1 + usage_c))
        got = float(mt.frequency_lp(a, p, eps, beta, scale, positions))
        assert abs(got - expected) <= 1e-9 * abs(expected), (got, expected)
    scales = [a.abs().mean() for a in parts]
    denominators = [mt.frequency_lp(a, p, eps, beta, sc, positions) for a, sc in zip(parts, scales)]
    lp = dict(p=p, epsilon=eps, beta=beta, scales=scales, positions=positions, denominators=denominators)
    assert abs(float(mt.lp_term(parts, lp)) - 1) < 1e-5            # each matrix over its own value (lp_term works in float32)
    x = parts[0].clone().requires_grad_()
    mt.frequency_lp(x, p, eps, beta, scales[0], positions).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    usage = [float(((a / sc + eps) ** p - eps ** p).mean(0).max()) * positions for a, sc in zip(parts, scales)]
    return dict(max_usage=usage, weight_at_max_usage=[1 + beta * math.log2(1 + u) for u in usage])


def check_certificate():
    """tail_certificate on small float64 pieces (two matrices, COUNTS swapped for the test): P_j, N_j, cert and quad
    against a loop over pieces; cert >= the brute-force worst case over every 0/1 restoration of the switched-off pieces
    of the sketched second-order KL, mean_j (sum_c v_c s_c^j)^2 / 2 (equal when K = 1), and quad is that KL at v = ablated;
    with a fractional mask the bound holds over a grid of fractional v; the gradient reaches reads and writes."""
    import itertools
    def effects(cache, pieces):     # s_c^j = (r_c.h)(u_c.g_j) by explicit loops over the input and output dimensions
        return torch.cat([(e['h'][:, :, None] * p.read).sum(1)[:, None, :] * (e['events']['cot'][..., None] * p.write).sum(2)
                          for e, p in zip(cache, pieces)], -1).detach()
    counts = mt.COUNTS
    g = torch.Generator().manual_seed(12)
    try:
        mt.COUNTS = [4, 5]
        for k_samples in (1, 4):
            n, shapes = 6, ((7, 3), (5, 8))                                     # (d_in, d_out) per matrix
            cache = [dict(kind='vjp', h=torch.randn(n, i, generator=g, dtype=torch.float64),
                          events=dict(cot=torch.randn(n, k_samples, o, generator=g, dtype=torch.float64))) for i, o in shapes]
            pieces = [core.FreePieces(torch.randn(i, c, generator=g, dtype=torch.float64).requires_grad_(),
                                      torch.randn(o, c, generator=g, dtype=torch.float64).requires_grad_())
                      for (i, o), c in zip(shapes, mt.COUNTS)]
            ablated = (torch.rand(n, sum(mt.COUNTS), generator=g) < .6).double()
            cert, quad = mt.tail_certificate(cache, pieces, 1, n, ablated[1:])
            s = effects(cache, pieces)[1:]                                      # [n-1, K, N]
            for t in range(n - 1):
                off = ablated[1 + t].bool()
                plus = s[t][:, off].clamp_min(0).sum(-1); minus = (-s[t][:, off]).clamp_min(0).sum(-1)
                assert abs(float(cert[t]) - float(torch.maximum(plus, minus).square().mean() / 2)) < 1e-10
                assert abs(float(quad[t]) - float((plus - minus).square().mean() / 2)) < 1e-10
                index = off.nonzero().flatten().tolist()
                worst = max(float((s[t][:, index] @ torch.tensor(v, dtype=torch.float64)).square().mean() / 2)
                            for v in itertools.product((0., 1.), repeat=len(index))) if index else 0.
                assert worst <= float(cert[t]) + 1e-10, (worst, float(cert[t]))
                if k_samples == 1: assert abs(worst - float(cert[t])) < 1e-10, (worst, float(cert[t]))
            (cert.sum() + quad.sum()).backward()
            assert all(torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for p in pieces for x in (p.read, p.write))
        # Fractional mask: switched-off fraction 1 - m in [0, 1]; any restoration v in [0, 1 - m] stays under the bound.
        frac = torch.rand(n, sum(mt.COUNTS), generator=g, dtype=torch.float64)
        cert, _ = mt.tail_certificate(cache, pieces, 0, n, frac)
        s_all = effects(cache, pieces)
        for _ in range(200):
            v = frac * torch.rand(frac.shape, generator=g, dtype=torch.float64).round()
            kl = ((s_all * v[:, None, :]).sum(-1).square().mean(-1) / 2)
            assert (kl <= cert.detach() + 1e-10).all()
    finally:
        mt.COUNTS = counts
    return 'ok'


def check_certificate_update(model, data, weights):
    """The certificate inside accumulate on 4 real contexts at 4 budgets (hard forward): each group's cert and quad match
    a float64 recomputation from the cache on the same hard keep set (bf16 tolerance); cert >= quad; with tau far above
    any cert the leaf gradients are those of the run without the term; with tau = 0 they change and stay finite."""
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:4]
    draws = list(DRAWS)
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
    run = lambda cert: mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=4, cert=cert)
    plain, _ = run(None)
    idle, idle_stats = run(dict(gamma=1., tau=1e9))
    active, active_stats = run(dict(gamma=1., tau=0.))
    flat = lambda leaves: torch.cat([t.grad.flatten() for l in leaves for t in (l.read, l.write)])
    base, same, moved = flat(plain), flat(idle), flat(active)
    result = dict(idle_grad_rel_diff=float((same - base).norm() / base.norm()), active_grad_rel_diff=float((moved - base).norm() / base.norm()))
    assert result['idle_grad_rel_diff'] < 1e-5 and result['active_grad_rel_diff'] > 1e-4 and torch.isfinite(moved).all(), result
    with torch.no_grad():
        s = mt.log_scores(cache, feasible, 0, len(rows)*CONTEXT)
        mask, _ = mt.forward_mask(s, draws, 'hard', CONTEXT)
        total = plus = 0.
        for e, f, m in zip(cache, feasible, (1 - mask).split(mt.COUNTS, -1)):
            effect = ((e['h'].double() @ f.read.double()) * m.double())[:, None, :] * (e['events']['cot'].double() @ f.write.double())
            total, plus = total + effect.sum(-1), plus + effect.clamp_min(0).sum(-1)
            del effect
        cert = torch.maximum(plus, plus - total).square().mean(-1).div(2).view(len(draws), -1).mean(-1)
        quad = total.square().mean(-1).div(2).view(len(draws), -1).mean(-1)
    result.update(cert=[x['cert'] for x in idle_stats], cert_quad=[x['cert_quad'] for x in idle_stats],
                  cert_float64=cert.tolist(), quad_float64=quad.tolist())
    for x, c, q in zip(idle_stats, cert.tolist(), quad.tolist()):
        assert abs(x['cert'] - c) <= 3e-2 * c and abs(x['cert_quad'] - q) <= 3e-2 * q + 1e-2 * c, result
        assert x['cert'] >= x['cert_quad'] and x['cert_hinge'] == 0., result
    assert all(abs(a['cert'] - b['cert']) <= 1e-6 * b['cert'] for a, b in zip(active_stats, idle_stats)), result
    return result


def check_mass(weights):
    """piece_mass is weight_mass67.py's per-matrix share (float64 reference), and the penalty's gradient reaches both
    factors of every matrix through the in-graph projection."""
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    norms = [float(w.double().pow(2).sum()) for w in weights]
    masses = mt.piece_mass(pieces, norms).double().cpu()
    exact = torch.tensor([float((p.read.double().pow(2).sum(0) * p.write.double().pow(2).sum(0)).sum()) / n for p, n in zip(pieces, norms)], dtype=torch.float64)
    assert torch.allclose(masses, exact, rtol=1e-4), (masses, exact)
    cfg = method.TrainConfig(precision='bf16', projection_precision='fp32')
    for p in pieces: p.read.requires_grad_(); p.write.requires_grad_()
    feasible = [method.project(w, p, cfg) for w, p in zip(weights, pieces)]
    penalty = mt.mass_penalty(feasible, norms)
    penalty.backward()
    for p in pieces:
        for t in (p.read, p.write): assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    return dict(initial_mass=[round(float(m), 3) for m in masses], initial_penalty=float(penalty))


def check_routing():
    """routing_masks is VPD's sample_uniform_k_subset_routing_masks bit for bit on the same generator; the number of
    masked matrices per position is uniform on 1..24."""
    from param_decomp.routing import sample_uniform_k_subset_routing_masks
    shape = (64, 512)
    ours = mt.routing_masks(torch.Generator().manual_seed(3), shape)
    theirs = sample_uniform_k_subset_routing_masks(shape, list(MODULES), 'cpu', torch.Generator().manual_seed(3))
    assert torch.equal(ours, torch.stack([theirs[n] for n in MODULES])), 'routing differs from VPD'
    counts = torch.bincount(ours.sum(0).flatten(), minlength=len(MODULES) + 1) / ours[0].numel()
    assert counts[0] == 0 and float((counts[1:] - 1 / len(MODULES)).abs().max()) < 0.01, counts
    return dict(masked_share=float(ours.float().mean()), expected=(len(MODULES) + 1) / 2 / len(MODULES))


def check_hard_frequency():
    """hard_frequency against a float64 loop over pieces; its gradient is the frequency weight passed straight through
    the hard mask to the soft mask's Jacobian; tail_share against a loop, with gradient only through the scores."""
    g = torch.Generator().manual_seed(12)
    s = (torch.randn(64, 300, generator=g, dtype=torch.float64) * 3).requires_grad_()
    mask, _ = mt.forward_mask(s, 17, 'hard')
    positions = 32768
    f = mask.detach().mean(0)
    expected = sum(float(x) * math.log2(1 + positions * float(x)) for x in f)
    got = mt.hard_frequency(mask, positions)
    assert abs(float(got) - expected) <= 1e-9 * expected, (float(got), expected)
    ours, = torch.autograd.grad(got, s)
    weight = (torch.log2(1 + positions * f) + positions * f / ((1 + positions * f) * math.log(2))) / mask.shape[0]
    theirs, = torch.autograd.grad((mt.sigmoid_topk(s, 17) * weight).sum(), s)
    assert torch.allclose(ours, theirs, atol=1e-12), float((ours - theirs).abs().max())
    a = [torch.rand(64, n, generator=g, dtype=torch.float64).requires_grad_() for n in (100, 200)]
    share = mt.tail_share(a, mask)
    whole = torch.cat(a, -1).detach()
    loop = [float(sum(whole[t, c] for c in range(300) if mask[t, c] == 0) / whole[t].sum()) for t in range(64)]
    assert torch.allclose(share.detach(), torch.tensor(loop, dtype=torch.float64), atol=1e-12)
    share.sum().backward()
    assert all(torch.isfinite(x.grad).all() for x in a) and s.grad is None
    return dict(hard_frequency=expected, tail_share_mean=float(share.mean()))


def check_mainline(model, data, weights):
    """The merged loss (MergedAdversary), on 4 real contexts with per-context budgets:
      - nothing attacked is the plain update (KL and gradients), and the sources do not move;
      - everything attacked at sources 0 with lr 0 is the plain update again;
      - everything attacked from uniform sources: the sources follow the added-loss Adversary (weight 1) step for step
        when both use the same Adam settings, the attacked KL is its adversarial KL, and the pieces' gradient is the
        added-loss gradient minus the plain one;
      - 2 particles: only the drawn particle moves, and the per-particle counts advance;
      - routing nothing masked gives the intact model; the same router seed gives the same update."""
    cfg = mt.vpd_adversary_config()
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:4]
    ids = rows[:, :-1]
    draws = list(DRAWS)
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
    grads = lambda leaves: [t.grad for l in leaves for t in (l.read, l.write)]
    result = {}
    plain, plain_stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=4)

    def merged(fraction, particles=1, zero=False, lr=.01, chunk=4, router=None):
        adversary = mt.MergedAdversary(cfg, fraction, particles, 4, seed=0)
        if zero: adversary.source.zero_()
        adversary.lr = lr
        adversary.draw(0)
        before = adversary.source.clone()
        leaves, stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=chunk,
                                      merged=adversary, router=router)
        return adversary, before, leaves, stats

    adversary, before, leaves, stats = merged(0.)
    assert not any(s['attacked'] for s in stats) and torch.equal(adversary.source, before)
    result['none_attacked'] = dict(kl_max_rel_diff=max(abs(a['kl'] - b['kl']) / b['kl'] for a, b in zip(stats, plain_stats)),
                                   grad_max_rel_diff=relative(grads(leaves), grads(plain)))
    assert result['none_attacked']['kl_max_rel_diff'] < 1e-6 and result['none_attacked']['grad_max_rel_diff'] < 1e-5, result
    adversary, before, leaves, stats = merged(1., zero=True, lr=0.)
    assert all(s['attacked'] for s in stats) and torch.equal(adversary.source, before)
    result['zero_sources'] = dict(kl_max_rel_diff=max(abs(a['kl'] - b['kl']) / b['kl'] for a, b in zip(stats, plain_stats)),
                                  grad_max_rel_diff=relative(grads(leaves), grads(plain)))
    assert result['zero_sources']['kl_max_rel_diff'] < 1e-6 and result['zero_sources']['grad_max_rel_diff'] < 1e-5, result
    assert int(adversary.count.sum()) == 4 * (cfg['n_warmup_steps'] + 1)
    # Against the added-loss Adversary (weight 1) from the same sources.
    adversary, before, leaves, stats = merged(1.)
    added = mt.Adversary(cfg, 1., 4, seed=0); added.source.copy_(before[0]); added.lr = .01
    added_leaves, added_stats = mt.accumulate(model, rows, cache, feasible, draws, 'hard', None, target_logp=logp, chunk=4, adversary=added)
    moved = (adversary.source[0] - added.source).abs()
    result['vs_added'] = dict(
        kl_max_rel_diff=max(abs(a['kl'] - b['adv_kl']) / b['adv_kl'] for a, b in zip(stats, added_stats)),
        start_max_rel_diff=max(abs(a['adv_kl_start'] - b['adv_kl_start']) / b['adv_kl_start'] for a, b in zip(stats, added_stats)),
        grad_max_rel_diff=relative(grads(leaves), [a - b for a, b in zip(grads(added_leaves), grads(plain))]),
        source_max_abs_diff=float(moved.max()), source_fraction_off_by_1e_4=float((moved > 1e-4).float().mean()),
        source_mean_move=float((adversary.source - before).abs().mean()), adv_kl=[s['kl'] for s in stats],
        adv_kl_start=[s['adv_kl_start'] for s in stats], plain_kl=[s['kl'] for s in plain_stats])
    assert result['vs_added']['kl_max_rel_diff'] < 1e-3 and result['vs_added']['start_max_rel_diff'] < 1e-6, result
    # SimpleStories: about 4% of entries move 1e-4 apart (Oct 2026): Adam steps on entries with near-zero gradients, where
    # bf16 rounding between the two code paths (ascent-gradient difference 0.5% overall) changes the normalized step.
    source_tolerance = 1e-2 if len(MODULES) == 24 else 6e-2
    assert result['vs_added']['grad_max_rel_diff'] < 5e-2 and result['vs_added']['source_fraction_off_by_1e_4'] < source_tolerance, result
    # Two particles: the drawn one moves, the other stays.
    adversary, before, _, _ = merged(1., particles=2)
    picked = torch.zeros(2, 4, dtype=torch.bool); picked[adversary.pick, torch.arange(4)] = True
    for p in range(2):
        for slot in range(4):
            same = torch.equal(adversary.source[p, slot], before[p, slot])
            assert same != bool(picked[p, slot]), (p, slot)
            assert int(adversary.count[p, slot]) == (cfg['n_warmup_steps'] + 1) * int(picked[p, slot])
    # Routing: masking no matrix is the intact model; masking all is the plain masked forward; a fixed seed repeats.
    keep = [(torch.rand(*ids.shape, c, device='cuda') < .5).float() for c in mt.COUNTS]
    with torch.no_grad():
        none = torch.zeros(len(MODULES), *ids.shape, dtype=torch.bool, device='cuda')
        kl_none = float(mt.kl_per_position(mt.intact_logp(model, ids), mt.masked_logits(model, ids, pieces, keep, none)).max())
        assert torch.equal(mt.masked_logits(model, ids, pieces, keep, ~none), mt.masked_logits(model, ids, pieces, keep))
    assert kl_none < 1e-3, kl_none
    seeded = lambda: torch.Generator(device='cuda').manual_seed(9)
    _, _, routed_a, stats_a = merged(.5, router=seeded())
    _, _, routed_b, stats_b = merged(.5, router=seeded())
    assert [s['kl'] for s in stats_a] == [s['kl'] for s in stats_b] and relative(grads(routed_a), grads(routed_b)) < 1e-4
    result['routing'] = dict(kl_unrouted_max=kl_none, routed_kl=[s['kl'] for s in stats_a], attacked=[s['attacked'] for s in stats_a])
    return result


def check_budgets():
    """The budget sampler: power 1, independent draws is the earlier runs' log-uniform draw, bit for bit; power 1.5,
    stratified puts exactly half of every update's draws on each side of the median and matches the density overall."""
    saved = mt.BUDGET_POWER, mt.BUDGET_STRATIFIED
    total = sum(mt.COUNTS)
    try:
        mt.BUDGET_POWER, mt.BUDGET_STRATIFIED = 1., False
        g_old, g_new = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        old = [min(total - 1, max(1, round(math.exp(math.log(mt.K_MIN) + float(torch.rand((), generator=g_old))
                                                    * (math.log(total - 1) - math.log(mt.K_MIN)))))) for _ in range(64)]
        assert mt.update_budgets(g_new, total, 64) == old
        mt.BUDGET_POWER, mt.BUDGET_STRATIFIED = 1.5, True
        g = torch.Generator().manual_seed(0)
        median, draws = mt.budget_at(0.5, total), []
        for _ in range(500):
            ks = mt.update_budgets(g, total, 64)
            assert sum(k < median for k in ks) <= 32 <= sum(k <= median for k in ks), ks
            draws += ks
        e = 1 - mt.BUDGET_POWER
        cdf = lambda k: (mt.K_MIN ** e - k ** e) / (mt.K_MIN ** e - (total - 1) ** e)
        below = sum(k < 100 for k in draws) / len(draws)
        assert abs(below - cdf(99.5)) < 0.01, (below, cdf(99.5))
        return dict(power=mt.BUDGET_POWER, median_budget=median, fraction_below_100=below, expected_below_100=cdf(99.5))
    finally:
        mt.BUDGET_POWER, mt.BUDGET_STRATIFIED = saved


def check_sigmoid_topk():
    g = torch.Generator().manual_seed(0)
    s = (torch.randn(5, 300, generator=g, dtype=torch.float64) * 3).requires_grad_()
    for k in (1, 17, 150, 299):
        alpha = mt.sigmoid_topk(s, k, iterations=80)
        assert torch.allclose(alpha.sum(-1), torch.full((5,), float(k), dtype=torch.float64), atol=1e-8), (k, alpha.sum(-1))
    # Jacobian of one row against diag(sigma') - sigma' sigma'^T / sum(sigma').
    row = s[:1].detach().clone()
    jac = torch.autograd.functional.jacobian(lambda x: mt.sigmoid_topk(x, 17, iterations=80), row)[0, :, 0, :]
    a = mt.sigmoid_topk(row, 17, iterations=80)[0]
    d = a * (1 - a)
    expected = torch.diag(d) - torch.outer(d, d) / d.sum()
    assert torch.allclose(jac, expected, atol=1e-10), float((jac - expected).abs().max())
    # Rows of the Jacobian sum to zero: the budget is exact for any perturbation.
    assert float(jac.sum(0).abs().max()) < 1e-10
    # Well-separated scores give the hard top-k mask.
    sharp = torch.tensor([[30., 20., 10., -10., -20., -30.]], dtype=torch.float64)
    assert torch.allclose(mt.sigmoid_topk(sharp, 3), torch.tensor([[1., 1., 1., 0., 0., 0.]], dtype=torch.float64), atol=1e-4)
    # Hard forward: exact 0/1 top-k values, the soft mask's gradient.
    hard, soft = mt.forward_mask(s, 17, 'hard')
    assert torch.equal(hard.detach().sum(-1), torch.full((5,), 17., dtype=torch.float64))
    assert torch.equal(hard.detach() > 0, s.detach() >= s.detach().topk(17, -1).values[:, -1:])
    weight = torch.randn(5, 300, generator=g, dtype=torch.float64)
    ours, = torch.autograd.grad((hard * weight).sum(), s)
    theirs, = torch.autograd.grad((mt.sigmoid_topk(s, 17) * weight).sum(), s)
    assert torch.allclose(ours, theirs, atol=1e-12)
    return float((jac - expected).abs().max())


def check_write_sensor(model, data, weights):
    """MATRYOSHKA_SENSOR=write: the cache's scores are |h r_c| ||u_c|| on each matrix's intact input, its log-probs are
    intact_logp's, and gradients reach both factors (read through h r, write through ||u||) of every matrix."""
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    rows = batch(data, 0)[:4]
    ids = rows[:, :-1]
    cache, logp = mt.capture(model, rows, seed=0, sensor='write', chunk=4, keep_logp=True)
    inputs = {}
    hooks = [model.get_submodule(n).register_forward_hook(lambda _m, i, _o, n=n: inputs.__setitem__(n, i[0])) for n in MODULES]
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            separate = mt.intact_logp(model, ids)
    finally:
        for h in hooks: h.remove()
    result = dict(logp_max_abs_diff=float((logp.view(*ids.shape, -1) - separate).abs().max()))
    errors = []
    for n, e, q in zip(MODULES, cache, pieces):
        h = inputs[n].flatten(0, 1).bfloat16().double()
        direct = (h @ q.read.double()).abs() * q.write.double().norm(dim=0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ours = method.structured.scores(e, q).double()
        errors.append(float(((ours - direct).abs() / direct.abs().clamp_min(1e-12)).median()))
    result['score_median_rel_diff_max'] = max(errors)
    assert result['logp_max_abs_diff'] < 1e-3 and result['score_median_rel_diff_max'] < 1e-2, result
    leaves = [core.FreePieces(q.read.detach().requires_grad_(), q.write.detach().requires_grad_()) for q in pieces]
    mt.log_scores(cache, leaves, 0, 4 * CONTEXT).logsumexp(-1).sum().backward()
    assert all(float(t.grad.abs().sum()) > 0 for q in leaves for t in (q.read, q.write)), 'gradient misses a factor'
    return result


def check_static_sensor(model, data, weights):
    """MATRYOSHKA_SENSOR=static: on the rows M is estimated from, the static score squared, averaged over positions, is
    the Fisher score squared averaged over the same positions and sampled labels (u^T M u = mean_x u^T F_x u, read
    factor aside, here with unit reads); the static entries score like the formula; M is PSD with L L^T = M."""
    import static_fisher
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    rows, k = 16, 4
    factors = static_fisher.estimate(model, data, rows=rows, k=k, seed=7, microbatch=16)
    block = torch.from_numpy(np.array(data[len(data)-rows:], dtype=np.int64)).cuda()
    cache = capture_fisher(model, block, k=k, seed=7, microbatch=16)
    result = {}
    for n, e, q, (L, values) in zip(MODULES, cache, pieces, factors):
        cot = e['events']['cot'].double()                                   # [positions, k, d_out]
        u = q.write.double()
        fisher = (cot @ u).square().mean((0, 1))                            # mean_x mean_k (g.u)^2 per piece
        static = (L.double().T @ u).square().sum(0)
        result[n] = float(((static - fisher).abs() / fisher.clamp_min(1e-30)).max())
        assert float(values.min()) >= 0
    result = dict(static_vs_fisher_mean_max_rel_diff=max(result.values()))
    assert result['static_vs_fisher_mean_max_rel_diff'] < 1e-3, result
    mt.STATIC_METRIC = factors
    try:
        static = mt.capture(model, block[:4], seed=0, sensor='static', chunk=4)
    finally:
        mt.STATIC_METRIC = None
    errors = []
    for e, q, (L, _) in zip(static, pieces, factors):
        direct = (e['h'].double() @ q.read.double()).abs() * (L.double().T @ q.write.double()).norm(dim=0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ours = method.structured.scores(e, q).double()
        errors.append(float(((ours - direct).abs() / direct.clamp_min(1e-12)).median()))
    result['score_median_rel_diff_max'] = max(errors)
    assert result['score_median_rel_diff_max'] < 2e-2, result
    return result


def check_tail_count():
    """tail_count: dropped pieces within TAIL_COUNT_FRACTION of the k-th kept score, per token."""
    import math as m
    s = torch.log(torch.tensor([[1., .5, .2, .09, .11, .04], [3., 2., 1., .3, .29, .5]]))
    mask = torch.tensor([[1., 1., 0., 0., 0., 0.], [1., 1., 1., 0., 0., 0.]])   # k-th kept: .5 and 1.
    count = mt.tail_count(s, mask)
    expected = torch.tensor([float(((torch.tensor([.2, .09, .11, .04]) >= .5 * mt.TAIL_COUNT_FRACTION)).sum()),
                             float(((torch.tensor([.3, .29, .5]) >= 1. * mt.TAIL_COUNT_FRACTION)).sum())])
    assert torch.equal(count, expected), (count, expected)
    return count.tolist()


def check_interact_score(model, data, weights):
    """metric_interact: per token and matrix, 2 v_c.y - ||v_c||^2 = ||y||^2 - ||y - v_c||^2 with v_c = a_c L^T u_c and
    y = sum_c v_c, checked by brute force in float64."""
    import static_fisher
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    factors = static_fisher.estimate(model, data, rows=16, k=4)
    rows = batch(data, 0)[:1]
    from write_attribution import capture_inputs
    cache = static_fisher.entries(capture_inputs(model, rows), factors, 'metric_interact')
    worst = 0.
    for e, q, (L, _) in zip(cache, pieces, factors):
        sub = dict(e, h=e['h'][:8])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ours = method.structured.scores(sub, q).double()
        a = sub['h'].double() @ q.read.double()                                   # [8, C]
        v = a[:, :, None] * (L.double().T @ q.write.double()).T[None]              # [8, C, rank]
        y = v.sum(1, keepdim=True)
        brute = y.square().sum(-1) - (y - v).square().sum(-1)
        worst = max(worst, float((ours - brute).abs().max() / brute.abs().max()))
    assert worst < 1e-3, worst
    return dict(interact_max_rel_diff=worst)


def check_greedy_scores(model, data, weights):
    """static_fisher.greedy_scores: on 4 tokens, each of the first 20 picks is the piece whose addition most shrinks
    sum_m ||y_m - kept_m||^2, recomputed from scratch in float64; the picks come first in the returned order."""
    import static_fisher
    from write_attribution import capture_inputs
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    factors = static_fisher.estimate(model, data, rows=16, k=4)
    cache = [dict(e, h=e['h'][:4]) for e in static_fisher.entries(capture_inputs(model, batch(data, 0)[:1]), factors)]
    scores = torch.cat(static_fisher.greedy_scores(cache, pieces, 20), -1)
    order = scores.argsort(-1, descending=True)[:, :20]
    vs = []                                                    # v_c for every piece, [4, N, rank_max] per matrix
    for e, q, (L, _) in zip(cache, pieces, factors):
        a = e['h'].double() @ q.read.double()
        vs.append(a[:, :, None] * (L.double().T @ q.write.double()).T[None])
    owner = torch.cat([torch.full((v.shape[1],), j) for j, v in enumerate(vs)])
    local = torch.cat([torch.arange(v.shape[1]) for v in vs])
    for t in range(4):
        residual = [v[t].sum(0) for v in vs]
        for step in range(20):
            gains = torch.cat([2 * (v[t] @ r) - v[t].square().sum(-1) for v, r in zip(vs, residual)])
            for c in order[t, :step].tolist(): gains[c] = float('-inf')
            best, chosen = gains.argmax().item(), order[t, step].item()
            assert abs(float(gains[best] - gains[chosen])) <= 1e-9 * float(gains[best].abs()), (t, step, best, chosen)
            residual[owner[chosen]] = residual[owner[chosen]] - vs[owner[chosen]][t, local[chosen]]
    return 'ok'


def check_masked_scorer(model, data, weights):
    """MaskedScorer (MATRYOSHKA_SCORE_INPUT=masked): every matrix keeps exactly the intact top-k's count per token; the
    layer-0 q/k/v inputs are untouched by masking, so their keep sets equal the intact ones; the whole budget keeps every
    piece (the target, up to bf16); gradients reach both factors of every matrix and the attack source."""
    import static_fisher
    from write_attribution import capture_inputs
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    factors = static_fisher.estimate(model, data, rows=16, k=4)
    old, mt.STATIC_METRIC = mt.STATIC_METRIC, factors
    try:
        rows = batch(data, 0)[:2]; ids = rows[:, :-1]
        cache = static_fisher.entries(capture_inputs(model, rows), factors)
        leaves = [core.FreePieces(p.read.detach().requires_grad_(), p.write.detach().requires_grad_()) for p in pieces]
        # Intact scores computed exactly as MaskedScorer does (bf16 matmul, fp32 product), so only the input can differ.
        factors_w = mt.write_factors(leaves)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            s = torch.cat([((e['h'] @ q.read).float().abs() * f).clamp_min(1e-30).log() for e, q, f in zip(cache, leaves, factors_w)], -1)
        mask, _ = mt.forward_mask(s, [71, 710], 'hard', CONTEXT)
        counts = mt.matrix_counts(mask)
        source = torch.rand(2 * CONTEXT, sum(mt.COUNTS), device='cuda').requires_grad_()
        scorer = mt.MaskedScorer(leaves, mt.write_factors(leaves), counts, list(source.split(mt.COUNTS, -1)))
        target_logp = mt.intact_logp(model, ids)
        kl = mt.kl_per_position(target_logp, mt.masked_logits(model, ids, leaves, scorer.alphas())).mean()
        s_m, keep = scorer.scores()
        assert torch.equal(mt.matrix_counts(keep), counts)
        for j in range(3):                      # h.0 q, k, v: the same sets (exact ties aside)
            a, b = keep.split(mt.COUNTS, -1)[j].detach().round(), mask.split(mt.COUNTS, -1)[j].detach().round()
            assert float((a != b).float().sum() / b.sum().clamp_min(1)) < 1e-3, (MODULES[j], float((a != b).float().sum()))
        changed = float((keep.detach() != mask.detach()).float().sum(-1).mean())
        (kl + scorer.parts[0].sum() * 0).backward()
        assert all(float(t.grad.abs().sum()) > 0 for q in leaves for t in (q.read, q.write)), 'gradient misses a factor'
        assert float(source.grad.abs().sum()) > 0
        full = mt.MaskedScorer(leaves, mt.write_factors(leaves), torch.tensor(mt.COUNTS, device='cuda').expand(2 * CONTEXT, -1))
        with torch.no_grad():
            kl_all = float(mt.kl_per_position(target_logp, mt.masked_logits(model, ids, leaves, full.alphas())).mean())
        assert kl_all < 0.05, kl_all
        return dict(kept_changed_per_token=changed, kl_all_kept=kl_all)
    finally:
        mt.STATIC_METRIC = old


def check_floors():
    """floored (floor 1, tau 0.1): a score at its floor maps to tau*log 2, one far above to score - floor, one far below
    to ~0; the floors get a gradient through it."""
    old = mt.FLOOR
    try:
        mt.FLOOR = torch.zeros(sum(mt.COUNTS), device='cuda').requires_grad_()       # floors at 1
        x = torch.full((3, sum(mt.COUNTS)), 1., device='cuda'); x[1] = 100.; x[2] = 1e-3
        out = torch.cat(mt.floored(list(x.split(mt.COUNTS, -1))), -1)
        assert abs(float(out[0, 0]) - (0.1 * math.log(2) + 1e-6)) < 1e-5 and abs(float(out[1, 0]) - 99) < 1e-3
        assert float(out[2, 0]) < 1e-5
        out.sum().backward()
        assert float(mt.FLOOR.grad.abs().sum()) > 0
        return 'ok'
    finally:
        mt.FLOOR = old


def check_floor_set(model, data, weights):
    """FLOOR_SET_FRACTION: a context drawn for the floor set trains at exactly its above-floor pieces (stats k = their
    mean count), the others at their budget; the floors get a gradient through the floor-set contexts."""
    import static_fisher
    old_metric, old_floor, old_ramp = mt.STATIC_METRIC, mt.FLOOR, mt.FLOOR_RAMP
    try:
        mt.FLOOR_RAMP = 0.                  # the hard floor set (the ramp's sizes are mask sums)
        mt.STATIC_METRIC = static_fisher.estimate(model, data, rows=16, k=4)
        pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
        rows = batch(data, 0)[:4]
        cache = mt.capture(model, rows, seed=0, sensor='static', chunk=4)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            med = torch.cat([method.structured.scores(e, p).float().median(0).values for e, p in zip(cache, pieces)])
        mt.FLOOR = med.clamp_min(1e-12).log().add(1.).requires_grad_()       # floors at e x median: a small set
        tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
        feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
        chosen = torch.tensor([True, False, True, False])
        leaves, stats = mt.accumulate(model, rows, cache, feasible, list(DRAWS), 'hard', None, chunk=4, floor_contexts=chosen)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = torch.cat([method.structured.scores(e, p).float() for e, p in zip(cache, feasible)], -1)
        sizes = (raw > mt.FLOOR.exp()).float().sum(-1).view(4, -1).mean(-1).round().clamp_min(1).long().tolist()
        assert [st['floor_set'] for st in stats] == chosen.tolist()
        assert [st['k'] for st in stats] == [sizes[0], DRAWS[1], sizes[2], DRAWS[3]], ([st['k'] for st in stats], sizes)
        assert mt.FLOOR.grad is not None and float(mt.FLOOR.grad.abs().sum()) > 0
        return dict(floor_set_sizes=[sizes[0], sizes[2]])
    finally:
        mt.STATIC_METRIC, mt.FLOOR, mt.FLOOR_RAMP = old_metric, old_floor, old_ramp


def check_attack_routing(model, data, weights):
    """ATTACK_ROUTING=all: with every context attacked, a routed update is the unrouted (all matrices masked) update."""
    cfg = mt.vpd_adversary_config()
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:4]
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
    out = []
    old = mt.ATTACK_ROUTING, mt.PLAIN_ROUTING
    try:
        for routed in (True, False):
            mt.ATTACK_ROUTING = 'all'
            adversary = mt.MergedAdversary(cfg, 1., 1, 4, seed=0); adversary.lr = .01; adversary.draw(0)
            router = torch.Generator(device='cuda').manual_seed(3) if routed else None
            out.append(mt.accumulate(model, rows, cache, feasible, list(DRAWS), 'hard', None, target_logp=logp, chunk=4,
                                     merged=adversary, router=router))
        plain = []                       # PLAIN_ROUTING=all with nothing attacked: the unrouted plain update
        for routed in (True, False):
            mt.ATTACK_ROUTING, mt.PLAIN_ROUTING = 'same', 'all'
            adversary = mt.MergedAdversary(cfg, 0., 1, 4, seed=0); adversary.draw(0)
            router = torch.Generator(device='cuda').manual_seed(3) if routed else None
            plain.append(mt.accumulate(model, rows, cache, feasible, list(DRAWS), 'hard', None, target_logp=logp, chunk=4,
                                       merged=adversary, router=router)[1])
    finally:
        mt.ATTACK_ROUTING, mt.PLAIN_ROUTING = old
    (a, sa), (b, sb) = out
    diff = max(abs(x['kl'] - y['kl']) / y['kl'] for x, y in zip(sa, sb))
    plain_diff = max(abs(x['kl'] - y['kl']) / y['kl'] for x, y in zip(*plain))
    assert diff < 1e-5 and plain_diff < 1e-5, (diff, plain_diff)
    return dict(kl_max_rel_diff=diff, plain_kl_max_rel_diff=plain_diff)


def check_plain_weight(model, data, weights):
    """PLAIN_WEIGHT: an attacked context's extra plain pass is its KL at the plain keep mask (the KL a plain context
    gets), and its gradient adds to the attacked one."""
    cfg = mt.vpd_adversary_config()
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:4]
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
    old = mt.PLAIN_WEIGHT
    try:
        mt.PLAIN_WEIGHT = 0.
        plain = mt.MergedAdversary(cfg, 0., 1, 4, seed=0); plain.draw(0)
        _, plain_stats = mt.accumulate(model, rows, cache, feasible, list(DRAWS), 'hard', None, target_logp=logp, chunk=4, merged=plain)
        mt.PLAIN_WEIGHT = 1.
        attacked = mt.MergedAdversary(cfg, 1., 1, 4, seed=0); attacked.lr = .01; attacked.draw(0)
        _, stats = mt.accumulate(model, rows, cache, feasible, list(DRAWS), 'hard', None, target_logp=logp, chunk=4, merged=attacked)
    finally:
        mt.PLAIN_WEIGHT = old
    diff = max(abs(a['plain_extra'] - b['kl']) / b['kl'] for a, b in zip(stats, plain_stats))
    assert diff < 1e-3, diff
    return dict(plain_extra_vs_plain_max_rel_diff=diff)


def check_global_source(model, data, weights):
    """ADV_GLOBAL_SHARE=1: every attacked context trains at the one global source; it moves, the slots' sources do not."""
    cfg = mt.vpd_adversary_config()
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    tc = method.TrainConfig(precision='bf16', projection_precision='fp32')
    rows = batch(data, 0)[:4]
    cache, logp = mt.capture(model, rows, seed=5, sensor='fisher', chunk=mt.MICRO, keep_logp=True)
    feasible = [method.project(w, p, tc) for w, p in zip(weights, pieces)]
    old = mt.GLOBAL_SHARE
    try:
        mt.GLOBAL_SHARE = 1.
        adversary = mt.MergedAdversary(cfg, 1., 1, 4, seed=0); adversary.lr = .01; adversary.draw(0)
        before, glob = adversary.source.clone(), adversary.glob.clone()
        _, stats = mt.accumulate(model, rows, cache, feasible, list(DRAWS), 'hard', None, target_logp=logp, chunk=4, merged=adversary)
    finally:
        mt.GLOBAL_SHARE = old
    assert bool(adversary.scoped.all()) and torch.equal(adversary.source, before)
    moved = float((adversary.glob - glob).abs().mean())
    assert moved > 0 and adversary.gcount == cfg['n_warmup_steps'] + 1, (moved, adversary.gcount)
    return dict(global_source_mean_move=moved)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    run_score_input, mt.SCORE_INPUT = mt.SCORE_INPUT, 'intact'    # the reference checks below are on intact scoring
    result = dict(sigmoid_topk_jacobian_max_error=check_sigmoid_topk(), bisection_kernel=check_bisection_kernel(),
                  budgets=check_budgets(), frequency=check_frequency(), certificate=check_certificate(),
                  routing=check_routing(), hard_frequency=check_hard_frequency(), tail_count=check_tail_count(), floors=check_floors())
    print(json.dumps(dict(budgets=result['budgets'], frequency=result['frequency'])), flush=True)
    model = target(); data, _ = tokens('train')
    weights = [model.get_submodule(n).weight for n in MODULES]
    result['speedups'] = check_speedups(model, data, weights)
    print(json.dumps(dict(speedups=result['speedups'], bisection_kernel=result['bisection_kernel'])), flush=True)
    result['adversary'] = check_adversary(model, data, weights)
    print(json.dumps(dict(adversary=result['adversary'])), flush=True)
    result['certificate_update'] = check_certificate_update(model, data, weights)
    print(json.dumps(dict(certificate_update=result['certificate_update'])), flush=True)
    result['mass'] = check_mass(weights)
    print(json.dumps(dict(mass=result['mass'])), flush=True)
    result['mainline'] = check_mainline(model, data, weights)
    result['write_sensor'] = check_write_sensor(model, data, weights)
    print(json.dumps(dict(write_sensor=result['write_sensor'])), flush=True)
    result['static_sensor'] = check_static_sensor(model, data, weights)
    print(json.dumps(dict(static_sensor=result['static_sensor'])), flush=True)
    result['interact_score'] = check_interact_score(model, data, weights)
    print(json.dumps(dict(interact_score=result['interact_score'])), flush=True)
    result['greedy_scores'] = check_greedy_scores(model, data, weights)
    result['masked_scorer'] = check_masked_scorer(model, data, weights)
    result['floor_set'] = check_floor_set(model, data, weights)
    result['attack_routing'] = check_attack_routing(model, data, weights)
    result['plain_weight'] = check_plain_weight(model, data, weights)
    result['global_source'] = check_global_source(model, data, weights)
    print(json.dumps(dict(floor_set=result['floor_set'])), flush=True)
    print(json.dumps(dict(masked_scorer=result['masked_scorer'])), flush=True)
    if mt.SENSOR == 'static':      # the checks below use the run's own sensor
        import static_fisher
        mt.STATIC_METRIC = static_fisher.estimate(model, data, rows=16, k=4)
    print(json.dumps(dict(mainline=result['mainline'])), flush=True)
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    rows = batch(data, 0)[:mt.MICRO]
    ids = rows[:, :-1]
    # All pieces kept: the factorized forward must reproduce the target (exact sum; bf16 matmuls leave a small floor).
    ones = [torch.ones(*ids.shape, c, device='cuda') for c in mt.COUNTS]
    with torch.no_grad():
        kl_all = float(mt.kl_per_position(mt.intact_logp(model, ids), mt.masked_logits(model, ids, pieces, ones)).mean())
    assert kl_all < 0.05, kl_all
    result['all_kept_kl'] = kl_all
    mt.SCORE_INPUT = run_score_input      # from here on, the run's own sensor and scoring input
    # One micro-batch: gradients reach both factors of all 24 matrices, then the projection.
    cfg = method.TrainConfig(precision='bf16', projection_precision='fp32')
    for p in pieces: p.read.requires_grad_(); p.write.requires_grad_()
    cache = mt.capture(model, rows, seed=5)       # the run's own sensor (MATRYOSHKA_SENSOR)
    result['sensor'] = mt.SENSOR
    feasible = [method.project(w, p, cfg) for w, p in zip(weights, pieces)]
    leaves, stats = mt.accumulate(model, rows, cache, feasible, [430])
    assert len(stats) == 1 and np.isfinite(stats[0]['kl']) and stats[0]['kl'] > 0
    for l in leaves:
        for t in (l.read, l.write):
            assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    torch.autograd.backward([t for f in feasible for t in (f.read, f.write)], [t.grad for l in leaves for t in (l.read, l.write)])
    for p in pieces:
        for t in (p.read, p.write):
            assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    result.update(micro_batch=stats[0])
    # Lp term: normalized by the same pieces on the same positions at p=1, it is exactly 1; gradients still flow.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        own = [method.structured.scores(method.structured.slice_entry(e, slice(0, len(rows)*512)), f).float() for e, f in zip(cache, feasible)]
    lp = dict(coeff=1., p=1., epsilon=1e-6, denominators=[method.lp_concentration(s, 1., 1e-6).mean() for s in own])
    lp_leaves, lp_stats = mt.accumulate(model, rows, cache, feasible, [430], lp=lp)
    # (masked scoring: the term is on the masked-input scores, so not exactly the intact normalizer)
    assert abs(lp_stats[0]['lp_normalized'] - 1) < 1e-3 or mt.SCORE_INPUT == 'masked', lp_stats[0]
    assert all(torch.isfinite(t.grad).all() for l in lp_leaves for t in (l.read, l.write))
    result['lp_check'] = lp_stats[0]['lp_normalized']
    # The masked forward with a hard mask is exactly evaluate_checkpoint.py's KLKeep replacement.
    with torch.no_grad():
        keep = [(torch.rand(*ids.shape, c, device='cuda') < .5).float() for c in mt.COUNTS]
        ours = mt.masked_logits(model, ids, pieces, keep)
        hooks = []
        for name, p, m in zip(MODULES, pieces, keep):
            hooks.append(model.get_submodule(name).register_forward_hook(
                lambda layer, inputs, output, p=p, m=m: ((inputs[0] @ p.read)*m) @ p.write.T))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            theirs = model(ids)[0]
        for h in hooks: h.remove()
    assert torch.equal(ours, theirs)
    # The held-out monitor runs end to end; keeping every piece is the all-kept forward again.
    curve = mt.monitor(model, rows, pieces, seed=3)
    assert all(np.isfinite(v) for v in curve.values()) and curve[sum(mt.COUNTS)] < 0.05, curve
    result['monitor'] = {str(k): v for k, v in curve.items()}
    print(json.dumps(dict(matryoshka_tests=result)), flush=True)
    print('MATRYOSHKA_TESTS_PASSED', flush=True)


if __name__ == '__main__': main()
