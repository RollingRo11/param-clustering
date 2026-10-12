"""Factor/selector/finite-response audit. Frozen 67M; Explorer only.

Binary endpoint search is per-context diagnostic optimization, not factor training.
All accepted moves reduce exact FP32 output KL at the fixed count per token.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/joint_select67_20260915'))
import joint_select67 as shared
import numpy as np
import torch
from common import MODULES, TARGET_SHA, target, tokens, save_json, sha, credentials
from fisher_attribution import capture_fisher
from gim_attribution import capture_gim
from train_pair import method

SPECS = {
    'vpd_published': ('vpd', 400000, 'checkpoints/published_vs_finished/vpd.pth'),
    'gim_5000': ('parauto', 5000, 'output/gim_1h/checkpoints/005000/model.pth'),
    'vpd_5000': ('vpd', 5000, 'checkpoints/step_5000/vpd.pth'),
    'gim_7783': ('parauto', 7783, 'output/gim_1h/checkpoints/007783/model.pth'),
}
shared.SPECS = SPECS
OUT = ROOT/'output/gap67_20260915'


def loss(logits, logp):
    return (logp.exp()*(logp-logits.float().log_softmax(-1))).sum(-1).mean()


def masks_of(keep, pieces):
    return [m[None] for m in keep.split([p.count for p in pieces], -1)]


@contextmanager
def intervention(model, pieces, masks, *, alpha=1., injected=None, layers=(0, 1, 2, 3)):
    hooks = []
    try:
        for i, (name, p, mask) in enumerate(zip(MODULES, pieces, masks, strict=True)):
            if int(name.split('.')[1]) not in layers:
                continue
            def hook(layer, inputs, output, p=p, mask=mask, i=i):
                if injected is not None:
                    return output+injected[i]
                value = ((inputs[0] @ p.read)*mask) @ p.write.T
                if layer.bias is not None:
                    value = value+layer.bias
                if isinstance(alpha, float) and alpha == 1.:
                    return value
                return output+alpha*(value-output)
            hooks.append(model.get_submodule(name).register_forward_hook(hook))
        yield
    finally:
        for hook in hooks:
            hook.remove()


def run(model, ids, pieces, keep, **kwargs):
    with intervention(model, pieces, masks_of(keep, pieces), **kwargs):
        return model(ids)[0]


@torch.no_grad()
def clean_capture(model, ids):
    inputs, hooks = {}, []
    try:
        for name in MODULES:
            def hook(_layer, args, _output, name=name):
                inputs[name] = args[0].detach()
            hooks.append(model.get_submodule(name).register_forward_hook(hook))
        baseline = model(ids)[0]
    finally:
        for hook in hooks:
            hook.remove()
    return baseline, inputs


def endpoint_search(model, ids, pieces, logp, initial, rounds):
    keep = initial.clone()
    with torch.no_grad():
        current = float(loss(run(model, ids, pieces, keep.float()), logp))
    history = [current]; trace = []
    for iteration in range(rounds):
        gate = keep.float().requires_grad_()
        value = loss(run(model, ids, pieces, gate), logp)
        grad, = torch.autograd.grad(value, gate)
        assert abs(float(value)-current) < 1e-5, (float(value), current)
        del gate, value
        # One pair per token; positive drop gradient and negative add gradient
        # are preferred. Candidate prefixes preserve every token's count.
        dval, drop = grad.masked_fill(~keep, -float('inf')).max(-1)
        aval, add = grad.masked_fill(keep, float('inf')).min(-1)
        predicted = aval-dval
        order = predicted.argsort()
        eligible = int((predicted < 0).sum())
        accepted = 0
        trials = []
        with torch.no_grad():
            for moves in sorted({min(n, eligible) for n in [512, 128, 32, 8, 1]}-{0}, reverse=True):
                t = order[:moves]
                candidate = keep.clone()
                candidate[t, drop[t]] = False
                candidate[t, add[t]] = True
                measured = float(loss(run(model, ids, pieces, candidate.float()), logp))
                trials.append(dict(swaps=moves, kl=measured))
                if measured < current-1e-6:
                    keep, current, accepted = candidate, measured, moves
                    break
        trace.append(dict(iteration=iteration, swaps=accepted, trials=trials))
        history.append(current)
        if accepted == 0:
            break
    assert torch.equal(keep.sum(-1), initial.sum(-1))
    return keep, dict(history=history, trace=trace,
                     changed_per_token=float((keep & ~initial).sum(-1).float().mean()))


def gradient_check(model, ids, pieces, logp, keep):
    gate = keep.float().requires_grad_()
    value = loss(run(model, ids, pieces, gate), logp)
    grad, = torch.autograd.grad(value, gate)
    direction = torch.zeros_like(gate)
    # A direction with a strong derivative gives a well-conditioned FP32 check.
    idx = grad.abs().flatten().topk(128).indices
    direction.flatten()[idx] = grad.flatten()[idx].sign()
    predicted = float((grad*direction).sum())
    del grad, value
    eps = 1e-3
    with torch.no_grad():
        plus = float(loss(run(model, ids, pieces, gate+eps*direction), logp))
        minus = float(loss(run(model, ids, pieces, gate-eps*direction), logp))
    observed = (plus-minus)/(2*eps)
    relative = abs(observed-predicted)/max(abs(predicted), 1e-6)
    assert relative < .03, (predicted, observed, relative)
    return dict(predicted=predicted, finite_difference=observed, relative_error=relative)


def diagnose(model, ids, pieces, keep, logp, inputs, *, exact=False):
    keep = keep.float()
    with torch.no_grad():
        actual = float(loss(run(model, ids, pieces, keep), logp))
        deltas, relative = [], []
        for name, p, mask in zip(MODULES, pieces, masks_of(keep, pieces), strict=True):
            h = inputs[name]
            clean = h @ model.get_submodule(name).weight.T
            delta = ((h @ p.read)*mask) @ p.write.T-clean
            deltas.append(delta)
            relative.append(float(delta.norm()/clean.norm().clamp_min(1e-12)))
        injected = float(loss(run(model, ids, pieces, keep, injected=deltas), logp))
        restored = float(loss(run(model, ids, pieces, keep, layers=(0, 1, 2)), logp))
    result = dict(kl=actual, clean_error_injection_kl=injected, restore_last_kl=restored,
                  clean_input_relative_error=relative)
    if exact:
        fn = lambda alpha: run(model, ids, pieces, keep, alpha=alpha)
        with torch.enable_grad():
            base, tangent = torch.autograd.functional.jvp(fn, torch.zeros((), device='cuda'),
                                                          torch.ones((), device='cuda'))
        with torch.no_grad():
            prob = base.softmax(-1)
            q = .5*((prob*tangent.square()).sum(-1)-(prob*tangent).sum(-1).square()).mean()
            result['exact_quadratic'] = float(q)
            result['path'] = {str(a): float(loss(fn(a), logp)) for a in [.01, .1, .25, .5, 1.]}
            result['jvp_base_kl'] = float(loss(base, logp))
        del base, tangent
    return result


@torch.no_grad()
def rank_and_census(cache, gim_cache, pieces):
    scores, gim_scores = [], []
    for e, ge, p in zip(cache, gim_cache, pieces, strict=True):
        read = e['h'].float() @ p.read
        effect = e['events']['cot'].float() @ p.write
        scores.append(read.abs()*(effect.square().mean(1)).sqrt())
        gim_scores.append((ge['h'].float() @ p.read).abs()*(ge['events']['g'].float() @ p.write).abs())
    scores = torch.cat(scores, -1)
    gs = torch.cat(gim_scores, -1)
    census = {}
    for label, values in [('fisher', scores), ('gim', gs)]:
        census[label] = {str(p): float(((values+1e-6).pow(p)-1e-6**p).sum(-1).mean()) for p in [.5, .641, 1.]}
        census[label]['soft_count_tau_001'] = float((values/(values+.001)).sum(-1).mean())
    return scores, census


@torch.no_grad()
def allocated_support(scores, pieces, target_masks):
    masks = []
    for s, target_mask in zip(scores.split([p.count for p in pieces], -1), target_masks, strict=True):
        needed = target_mask.sum(-1).long()
        assert int(needed.max()) <= s.shape[-1]
        order = s.argsort(-1, descending=True)
        ranked = torch.arange(s.shape[-1], device='cuda')[None] < needed[:, None]
        masks.append(torch.zeros_like(ranked).scatter_(-1, order, ranked))
    return torch.cat(masks, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first', type=int, default=424)
    ap.add_argument('--sequences', type=int, default=8)
    ap.add_argument('--rounds', type=int, default=32)
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    credentials(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    model = target(); data, manifest = tokens('val')
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(**vars(args), checkpoints=SPECS, budgets=[256, 512], fisher_k=128,
                source_sha256=sha(__file__), target_sha256=TARGET_SHA, val_sha256=manifest['val']['sha256'],
                precision='FP32 forwards and gradients; BF16 intact Fisher/GIM cache only',
                search='binary per-token swaps, actual KL acceptance, magnitude and jitter starts',
                parameters=sum(p.numel() for p in model.parameters()))
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text()) == json.loads(json.dumps(plan))
    else:
        save_json(plan, OUT/'plan.json')
    deadline = float(os.environ.get('GAP_DEADLINE', 'inf'))
    loaded = {name: shared.load(model, name) for name in SPECS}
    for first in range(args.first, args.first+args.sequences):
        if all((OUT/f'{first}_{name}_{k}.json').exists() for name in SPECS for k in plan['budgets']):
            continue
        if time.time() > deadline:
            return 75
        rows = torch.tensor(np.array(data[first:first+1], dtype=np.int64), device='cuda')
        ids = rows[:, :-1]
        baseline, inputs = clean_capture(model, ids)
        logp = baseline.log_softmax(-1).detach()
        cache = capture_fisher(model, rows, k=128, seed=6_100_000+first, microbatch=1)
        gim_cache = capture_gim(model, rows, microbatch=1)
        native = {}
        for name, (pieces, trainer, metadata) in loaded.items():
            if trainer is not None:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    base = trainer.model(ids, cache_type='input')
                    ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                    native[name] = torch.cat([ci[n].float()[0] for n in MODULES], -1).argsort(dim=-1, descending=True, stable=True)
                    del base, ci
        for name, (pieces, trainer, metadata) in loaded.items():
            if all((OUT/f'{first}_{name}_{k}.json').exists() for k in plan['budgets']):
                continue
            scores, census = rank_and_census(cache, gim_cache, pieces)
            order = scores.argsort(dim=-1, descending=True, stable=True)
            gen = torch.Generator(device='cuda').manual_seed(7_100_000+first)
            jitter = scores*torch.exp(.5*torch.randn(scores.shape, device='cuda', generator=gen))
            jitter_order = jitter.argsort(dim=-1, descending=True, stable=True)
            del jitter
            for budget in plan['budgets']:
                dest = OUT/f'{first}_{name}_{budget}.json'
                if dest.exists():
                    continue
                if time.time() > deadline:
                    return 75
                began = time.perf_counter()
                magnitude = shared.support(order, budget)
                record = dict(first_sequence=first, method=name, budget=budget, **metadata,
                              job_id=os.environ['SLURM_JOB_ID'], census=census, variants={})
                saved_masks = {'magnitude': magnitude}
                if not (OUT/'gradient_check.json').exists():
                    save_json(gradient_check(model, ids, pieces, logp, magnitude), OUT/'gradient_check.json')
                # Exact tangents on first two contexts; finite-response controls on all eight.
                exact = first < args.first+2
                record['variants']['magnitude'] = diagnose(model, ids, pieces, magnitude, logp, inputs, exact=exact)
                for label, start_order in [('endpoint_from_magnitude', order), ('endpoint_from_jitter', jitter_order)]:
                    optimized, history = endpoint_search(model, ids, pieces, logp, shared.support(start_order, budget), args.rounds)
                    record['variants'][label] = dict(**diagnose(model, ids, pieces, optimized, logp, inputs, exact=exact), search=history)
                    saved_masks[label] = optimized
                labels = ['magnitude', 'endpoint_from_magnitude', 'endpoint_from_jitter']
                record['best_endpoint'] = min(labels, key=lambda s: record['variants'][s]['kl'])
                if name in native:
                    selected = shared.support(native[name], budget)
                    record['variants']['native'] = diagnose(model, ids, pieces, selected, logp, inputs, exact=exact)
                    saved_masks['native'] = selected
                else:
                    vp = loaded['vpd_published'][0]
                    vmask = shared.support(native['vpd_published'], budget)
                    allocated = allocated_support(scores, pieces, vmask.split([p.count for p in vp], -1))
                    assert torch.equal(allocated.sum(-1), magnitude.sum(-1))
                    record['variants']['vpd_allocation'] = diagnose(model, ids, pieces, allocated, logp, inputs, exact=exact)
                    saved_masks['vpd_allocation'] = allocated
                with torch.no_grad():
                    all_keep = torch.ones_like(magnitude, dtype=torch.float32)
                    record['all_kept_kl'] = float(loss(run(model, ids, pieces, all_keep), logp))
                sp = OUT/f'{first}_{name}_{budget}_supports.npz'
                np.savez_compressed(sp, **{label: mask.nonzero()[:, 1].reshape(512, budget).cpu().numpy().astype(np.uint16)
                                          for label, mask in saved_masks.items()})
                record['support_sha256'] = sha(sp)
                record['seconds'] = time.perf_counter()-began
                save_json(record, dest)
                print(json.dumps(dict(sequence=first, method=name, budget=budget, seconds=record['seconds'],
                                      kl={n:v['kl'] for n,v in record['variants'].items()})), flush=True)
            del scores, order, jitter_order
        del cache, gim_cache, baseline, logp, inputs
    from gap67_report import finish
    finish(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
