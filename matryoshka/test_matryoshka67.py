"""Checks for matryoshka67.py: the sigmoid top-k threshold and its implicit gradient, the all-kept
masked forward, gradient flow through one micro-batch into every matrix's free factors, the sped-up
update (Triton bisection, topk threshold, chunked micro-batches, cached KL target) against the
one-micro-batch-at-a-time implementation it replaced (reference_* below, as of commit 3c2ccc6), the
budget sampler (power, stratification, one budget per context), and the persistent PGD loss against VPD's own code."""
import json
import math

import numpy as np
import torch

import matryoshka67 as mt
from common import MODULES, CONTEXT, target, tokens, batch
from fisher_attribution import capture_fisher
from train_pair import core, method


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
    draws = [40, 430, 4301, 30000]
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
    assert result['cache_chunk16_vs_4']['h_max_rel_diff'] < 1e-2 and result['cache_chunk16_vs_4']['cot_max_rel_diff'] < 5e-2, result
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
    draws = [40, 430, 4301, 30000]
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


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(sigmoid_topk_jacobian_max_error=check_sigmoid_topk(), bisection_kernel=check_bisection_kernel(),
                  budgets=check_budgets())
    print(json.dumps(dict(budgets=result['budgets'])), flush=True)
    model = target(); data, _ = tokens('train')
    weights = [model.get_submodule(n).weight for n in MODULES]
    result['speedups'] = check_speedups(model, data, weights)
    print(json.dumps(dict(speedups=result['speedups'], bisection_kernel=result['bisection_kernel'])), flush=True)
    result['adversary'] = check_adversary(model, data, weights)
    print(json.dumps(dict(adversary=result['adversary'])), flush=True)
    result['mass'] = check_mass(weights)
    print(json.dumps(dict(mass=result['mass'])), flush=True)
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, mt.COUNTS)]
    rows = batch(data, 0)[:mt.MICRO]
    ids = rows[:, :-1]
    # All pieces kept: the factorized forward must reproduce the target (exact sum; bf16 matmuls leave a small floor).
    ones = [torch.ones(*ids.shape, c, device='cuda') for c in mt.COUNTS]
    with torch.no_grad():
        kl_all = float(mt.kl_per_position(mt.intact_logp(model, ids), mt.masked_logits(model, ids, pieces, ones)).mean())
    assert kl_all < 0.05, kl_all
    result['all_kept_kl'] = kl_all
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
    assert abs(lp_stats[0]['lp_normalized'] - 1) < 1e-3, lp_stats[0]
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
