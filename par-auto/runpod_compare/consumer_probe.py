"""Decisive test for consumer-gated importance: does it recover what VPD's gate network learned?

At a fixed budget, rank pieces three ways on the SAME contexts with the SAME masking code and compare
the actual KL and CE of the masked model:
  - the method's own sensor (VPD: causal importance; par-auto: Fisher K sampled labels),
  - Fisher K sampled labels for both,
  - consumer-gated importance (consumer_attribution.py), seeded and q/k-filled by the Fisher scores,
  - the signed sketch at the intact point (`signed`) and at the sparse point (`sparse`): same probes,
    same rules, same on-set iteration, but `sparse` recaches activations, slopes, patterns and
    cotangents from the masked forward each round (sparse_sketch.iterate_sparse).
Published VPD is the case that matters: its own gate gives a far better support than any intact-point
gradient sensor, so a sensor that closes part of that distance recovers gate-like information without
learning a gate.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method
import consumer_attribution as consumer
import sparse_sketch

SPECS = {
    'vpd_published': ('vpd', 400000, 'checkpoints/published_vs_finished/vpd.pth'),
    'vpd_5000': ('vpd', 5000, 'checkpoints/step_5000/vpd.pth'),
    'gim_5000': ('parauto', 5000, 'output/gim_1h/checkpoints/005000/model.pth'),
    'fisher_k4_5000': ('parauto', 5000, 'output/fisher_k4_5000/checkpoints/005000/model.pth'),
    'parauto_1b': ('parauto', 30518, 'checkpoints/published_vs_finished/parauto.pth'),
}


def load(model, name):
    kind, step, rel = SPECS[name]
    path = ROOT/rel
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['method'] == kind and saved['completed'] == step and saved['target_sha256'] == TARGET_SHA
    trainer = None
    if kind == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True)
        handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors'])
        trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(),
                                  trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda().float(), p['write'].cuda().float()) for p in saved['factors']]
    return pieces, trainer, dict(checkpoint=str(path), checkpoint_sha256=sha(path), kind=kind, step=step)


def self_test():
    """Pure algebra checks on the backward recursion, no model."""
    n = 4
    counts = {'h.0.attn.o_proj': 2, 'h.1.attn.v_proj': 3}
    direct = {'h.0.attn.o_proj': torch.tensor([1., 2.]), 'h.1.attn.v_proj': torch.zeros(3)}
    on = {k: torch.ones(n, c) for k, c in counts.items()}
    scales = {k: torch.ones(n, 1) for k in counts}
    order = ['h.1.attn.v_proj', 'h.0.attn.o_proj']

    got = consumer.propagate(direct, {}, on, scales, counts, order)
    assert torch.allclose(got['h.0.attn.o_proj'], direct['h.0.attn.o_proj'].expand(n, 2)), got
    assert torch.allclose(got['h.1.attn.v_proj'], torch.zeros(n, 3))
    print('no consumers: importance is the direct term alone, and zero without one')

    # A chain: the v pieces of layer 1 are read by the o pieces of layer 0 in this toy graph.
    coupling = torch.tensor([[.5, 0., 1.], [0., 2., 1.]])            # [C_reader=2, C_producer=3]
    couplings = {('h.0.attn.o_proj', 'h.1.attn.v_proj'): coupling}
    order = ['h.0.attn.o_proj', 'h.1.attn.v_proj']
    got = consumer.propagate(direct, couplings, on, scales, counts, order)
    want = direct['h.0.attn.o_proj'][None, :] @ coupling
    assert torch.allclose(got['h.1.attn.v_proj'], want.expand(n, 3)), (got['h.1.attn.v_proj'], want)
    print(f'chain: producer importance flows from its on consumers through |K| ({want.tolist()})')

    off = {k: v.clone() for k, v in on.items()}
    off['h.0.attn.o_proj'][:] = 0.
    got = consumer.propagate(direct, couplings, off, scales, counts, order)
    assert torch.allclose(got['h.1.attn.v_proj'], torch.zeros(n, 3)), got['h.1.attn.v_proj']
    print('hard gate: with every consumer off, the producer scores exactly zero')

    half = {k: v.clone() for k, v in on.items()}
    half['h.0.attn.o_proj'][:, 1] = 0.
    got = consumer.propagate(direct, couplings, half, scales, counts, order)
    want = direct['h.0.attn.o_proj'][None, :1] @ coupling[:1]
    assert torch.allclose(got['h.1.attn.v_proj'], want.expand(n, 3))
    print('partial gate: only the surviving consumer contributes')

    # The RMSNorm scale applies only where the consumer reads the residual stream, i.e. for a
    # residual-writing producer. Here the producer is v_proj, read inside its own block, so tripling
    # the scales must change nothing.
    scaled = {k: v*3 for k, v in scales.items()}
    plain = consumer.propagate(direct, couplings, on, scales, counts, order)
    tripled = consumer.propagate(direct, couplings, on, scaled, counts, order)
    assert torch.allclose(plain['h.1.attn.v_proj'], tripled['h.1.attn.v_proj'])
    # With a residual writer as the producer, the same change must scale the carried importance by 3.
    stream = {('h.1.attn.v_proj', 'h.0.attn.o_proj'): torch.ones(3, 2)}
    flipped = ['h.1.attn.v_proj', 'h.0.attn.o_proj']
    a = consumer.propagate(direct, stream, on, scales, counts, flipped)['h.0.attn.o_proj']
    b = consumer.propagate(direct, stream, on, scaled, counts, flipped)['h.0.attn.o_proj']
    assert torch.allclose(b - direct['h.0.attn.o_proj'], 3*(a - direct['h.0.attn.o_proj']))
    print('norm scale is carried for residual writers only, and scales their term exactly')
    injected = {'h.0.attn.o_proj': torch.full((n, 2), 7.)}
    got = consumer.propagate(direct, couplings, on, scales, counts, order, injected)
    assert torch.allclose(got['h.0.attn.o_proj'], injected['h.0.attn.o_proj'])
    print('injected importance overrides the recursion for unmodelled sites')
    print('CONSUMER_SELF_TEST_PASSED')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--verify', action='store_true', help='finite-difference check of the signed sketch effects')
    ap.add_argument('--verify-sparse', action='store_true',
                    help='finite-difference check of the sketch at a sparse on-set, against the MASKED forward')
    ap.add_argument('--verify-budget', type=int, default=512, help='on-set size per token for --verify-sparse')
    ap.add_argument('--modes', nargs='+', default=['signed'], help="plus the base sensor, always included")
    ap.add_argument('--name', choices=sorted(SPECS))
    ap.add_argument('--budgets', type=int, nargs='+', default=[512])
    ap.add_argument('--sequences', type=int, default=16)
    ap.add_argument('--first-row', type=int, default=384)
    ap.add_argument('--fisher-k', type=int, default=64)
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--sketch-k', type=int, default=8)
    ap.add_argument('--sensor', choices=['gim', 'fisher'], default='gim',
                    help='scores that seed the on-set and fill q/k for the |K| mode, and the rules the signed sketch uses')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    assert args.name and args.output
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    from fisher_attribution import capture_fisher
    started = time.perf_counter()
    model = target()
    pieces, trainer, meta = load(model, args.name)
    counts = [p.count for p in pieces]
    data, _ = tokens('val')
    if args.verify:
        verify(model, pieces, torch.tensor(np.array(data[args.first_row:args.first_row+2], dtype=np.int64), device='cuda'))
        return
    if args.verify_sparse:
        rows = torch.tensor(np.array(data[args.first_row:args.first_row+2], dtype=np.int64), device='cuda')
        cache = capture_fisher(model, rows, k=args.fisher_k, seed=args.first_row)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            fallback = [method.scores_of(e, p).float() for e, p in zip(cache, pieces)]
        del cache
        verify_sparse(model, pieces, rows, fallback, args.verify_budget)
        return
    graph = None
    if 'abs' in args.modes:
        calibration = [torch.tensor(np.array(data[900+2*j:902+2*j], dtype=np.int64), device='cuda') for j in range(2)]
        graph = consumer.build_graph(model, pieces, calibration, fisher_k=8)
        del calibration
    rankings = [args.sensor] + [m for m in args.modes if m != args.sensor] + (['native'] if meta['kind'] == 'vpd' else [])
    sums = {r: {b: dict(kl=0., ce=0., kept=0.) for b in args.budgets} for r in rankings}
    baseline_ce, tokens_seen, churn, agreement = 0., 0, {}, None
    sketch_rules = 'gim' if args.sensor == 'gim' else 'exact'

    for first in range(args.first_row, args.first_row + args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda')
        ids, labels = rows[:, :-1], rows[:, 1:]
        if args.sensor == 'gim':
            from gim_attribution import capture_gim
            cache = capture_gim(model, rows)
        else:
            cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            base_scores = [method.scores_of(e, p).float() for e, p in zip(cache, pieces)]
        del cache
        fisher = base_scores                     # seeds the on-set and fills q/k in the |K| mode
        scores = {args.sensor: base_scores}
        if 'native' in rankings:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores['native'] = [ci.lower_leaky[n].float().flatten(0, 1) for n in MODULES]
                del base, ci
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            baseline = model(ids)[0]
            logp = baseline.float().log_softmax(-1); prob = logp.exp()
            baseline_ce += float(F.cross_entropy(baseline.flatten(0, 1), labels.flatten(), reduction='sum'))
            del baseline
        tokens_seen += ids.numel()
        if 'contrast' in args.modes:
            from contrast_attribution import contrast_scores
            contrast, contrast_info = contrast_scores(model, rows, pieces, seed=first)
            scores['contrast'] = contrast
        circuit = (sparse_sketch.Circuit(model, rows, pieces, probe_k=args.sketch_k, seed=first, rules=sketch_rules)
                   if ('signed' in args.modes or 'sparse' in args.modes) else None)
        for budget in args.budgets:
            if 'abs' in args.modes:
                gated, info = consumer.consumer_scores(model, rows, pieces, graph, budget, fisher,
                                                       iterations=args.iterations)
                scores['abs'] = gated
                churn.setdefault('abs', []).append(info['on_set_churn'])
            if 'signed' in args.modes:
                signed, info = sparse_sketch.iterate(circuit, budget, fisher, iterations=args.iterations)
                scores['signed'] = signed
                churn.setdefault('signed', []).append(info['on_set_churn'])
            if 'sparse' in args.modes:
                if meta['kind'] == 'vpd' and agreement is None:
                    agreement = masked_forward_agreement(model, trainer, pieces, ids, fisher, budget)
                    print(json.dumps(dict(vpd_masked_forward_agreement=agreement)), flush=True)
                sparse, info = sparse_sketch.iterate_sparse(model, rows, pieces, budget, fisher, iterations=args.iterations,
                                                            probe_k=args.sketch_k, seed=first, rules=sketch_rules,
                                                            labels=circuit.labels)
                scores['sparse'] = sparse
                churn.setdefault('sparse', []).append(info['on_set_churn'])
            for ranking in rankings:
                combined = torch.cat(scores[ranking], -1)
                order = combined.argsort(dim=-1, descending=True, stable=True)
                keep = torch.zeros_like(combined).scatter_(-1, order[..., :budget], 1.)
                masks = dict(zip(MODULES, [m.reshape(*ids.shape, -1) for m in keep.split(counts, -1)]))
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if meta['kind'] == 'vpd':
                        from param_decomp.models.components import make_mask_infos
                        pred = trainer.model(ids, mask_infos=make_mask_infos(masks))
                    else:
                        hooks = []
                        try:
                            for name, p in zip(MODULES, pieces):
                                def replace(layer, inputs, output, p=p, mask=masks[name]):
                                    value = ((inputs[0] @ p.read)*mask) @ p.write.T
                                    return value if layer.bias is None else value+layer.bias
                                hooks.append(model.get_submodule(name).register_forward_hook(replace))
                            pred = model(ids)[0]
                        finally:
                            for hook in hooks: hook.remove()
                    sums[ranking][budget]['kl'] += float((prob*(logp-pred.float().log_softmax(-1))).sum())
                    sums[ranking][budget]['ce'] += float(F.cross_entropy(pred.flatten(0, 1), labels.flatten(), reduction='sum'))
                    sums[ranking][budget]['kept'] += float(keep.sum())
                del pred, masks, keep, order, combined
        del scores, fisher, logp, prob, circuit

    result = dict(name=args.name, provenance=meta, sequences=args.sequences, first_row=args.first_row,
                  fisher_k=args.fisher_k, iterations=args.iterations, tokens=tokens_seen,
                  baseline_ce=baseline_ce/tokens_seen, graph=(graph['provenance'] if graph else None),
                  on_set_churn={mode: [float(np.mean([c[i] for c in per if i < len(c)]))
                                       for i in range(max((len(c) for c in per), default=0))]
                                for mode, per in churn.items()},
                  vpd_masked_forward_agreement=agreement,
                  results={r: {str(b): dict(kl=v['kl']/tokens_seen, ce=v['ce']/tokens_seen,
                                            ce_increase=(v['ce']-baseline_ce)/tokens_seen,
                                            kept_per_token=v['kept']/tokens_seen)
                               for b, v in per.items()} for r, per in sums.items()},
                  seconds=round(time.perf_counter()-started, 1), source_sha256=sha(__file__))
    save_json(result, args.output)
    print(json.dumps({k: result[k] for k in ('name', 'results', 'on_set_churn', 'seconds')}, indent=1), flush=True)



def masked_forward_agreement(model, trainer, pieces, ids, fallback, keep):
    """VPD's own masked forward (mask_infos, no weight delta) against the hook-based masked forward the
    sparse circuit is cached under, at the fallback on-set. Both bf16; reports how far apart they are."""
    from param_decomp.models.components import make_mask_infos
    counts = [p.count for p in pieces]
    on, _ = sparse_sketch.on_set(fallback, keep, ids.shape[0], ids.shape[1], counts)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        theirs = trainer.model(ids, mask_infos=make_mask_infos(on)).float()
        hooks = sparse_sketch.masked_hooks(model, pieces, on)
        try:
            ours = model(ids)[0].float()
        finally:
            for hook in hooks: hook.remove()
    lp_theirs, lp_ours = theirs.log_softmax(-1), ours.log_softmax(-1)
    return dict(keep=int(keep), max_abs_logit_diff=float((theirs-ours).abs().max()),
                mean_abs_logit_diff=float((theirs-ours).abs().mean()),
                kl_theirs_ours=float((lp_theirs.exp()*(lp_theirs-lp_ours)).sum(-1).mean()))


def verify_sparse(model, pieces, rows, fallback, keep, per_site=2, epsilons=(.25, .05, .01, .002), seed=0):
    """At a sparse on-set the circuit IS the masked model, so the predicted first-order effect of
    removing one ON piece, or inserting one OFF piece, at one position must match a finite
    difference of the MASKED forward pass. Same protocol as `verify`: FP32, loudest pieces per site.
    """
    counts = [p.count for p in pieces]
    B, T = rows.shape[0], rows.shape[1]-1
    on, _ = sparse_sketch.on_set(fallback, keep, B, T, counts)
    circuit = sparse_sketch.Circuit(model, rows, pieces, probe_k=4, seed=seed, rules='exact', autocast=False, masks=on)
    effects = sparse_sketch.backward_effects(circuit, on)
    ids = rows[:, :-1]

    def probe_losses(extra=None):
        hooks = sparse_sketch.masked_hooks(model, pieces, on)
        try:
            if extra is not None: hooks.append(extra())
            with torch.no_grad():
                logits = model(ids)[0].flatten(0, 1).float()
                return torch.stack([F.cross_entropy(logits, circuit.labels[:, j], reduction='sum')
                                    for j in range(circuit.probe_k)]).float()
        finally:
            for hook in hooks: hook.remove()

    base = probe_losses()
    with torch.no_grad():
        intact = model(ids)[0].flatten(0, 1).float().log_softmax(-1)
        hooks = sparse_sketch.masked_hooks(model, pieces, on)
        try:
            masked = model(ids)[0].flatten(0, 1).float().log_softmax(-1)
        finally:
            for hook in hooks: hook.remove()
        print(json.dumps(dict(keep=int(keep), kept_per_token=float(sum(m.sum() for m in on.values())/(B*T)),
                              masked_kl=float((intact.exp()*(intact-masked)).sum(-1).mean()))), flush=True)
        del intact, masked
    results = []
    for site in MODULES:
        index = MODULES.index(site)
        rho = circuit.activation[site].view(B, T, -1)
        size = rho.abs() * torch.linalg.vector_norm(effects[site], dim=2)                # [B,T,C]
        for direction, selector in (('remove', on[site]), ('insert', 1.-on[site])):
            sign = -1. if direction == 'remove' else 1.
            flat = (size*selector).flatten()
            for choice in flat.topk(per_site).indices.tolist():
                if float(flat[choice]) <= 0.: continue
                b, rest = divmod(choice, T*size.shape[-1])
                t, piece = divmod(rest, size.shape[-1])
                assert float(on[site][b, t, piece]) == (1. if direction == 'remove' else 0.)
                value = float(rho[b, t, piece])
                write = pieces[index].write.float()[:, piece]
                for epsilon in epsilons:
                    predicted = sign*epsilon*value*effects[site][b, t, :, piece]
                    def perturb(_layer, _inputs, output, b=b, t=t, value=value, write=write, epsilon=epsilon, sign=sign):
                        changed = output.clone()
                        changed[b, t] = changed[b, t] + sign*epsilon*value*write.to(changed.dtype)
                        return changed
                    make = lambda site=site, perturb=perturb: model.get_submodule(site).register_forward_hook(perturb)
                    actual = probe_losses(make) - base
                    scale = max(float(predicted.norm()), float(actual.norm()), 1e-30)
                    results.append(dict(site=site, direction=direction, piece=piece, position=t, epsilon=epsilon, rho=value,
                                        predicted=float(predicted.norm()), actual=float(actual.norm()),
                                        ratio=float(actual.norm()/max(float(predicted.norm()), 1e-30)),
                                        relative_error=float((predicted-actual).norm()/scale),
                                        cosine=float(F.cosine_similarity(predicted[None], actual[None])[0])))
                    print(json.dumps(results[-1]), flush=True)
    for direction in ('remove', 'insert'):
        for epsilon in epsilons:
            subset = [r for r in results if r['epsilon'] == epsilon and r['direction'] == direction and r['predicted'] > 1e-3]
            if not subset: continue
            print(json.dumps(dict(direction=direction, epsilon=epsilon, samples=len(subset),
                                  worst_relative_error=max(r['relative_error'] for r in subset),
                                  median_ratio=float(np.median([r['ratio'] for r in subset])),
                                  min_cosine=min(r['cosine'] for r in subset))), flush=True)
    print('SPARSE_VERIFY_DONE', flush=True)
    return results


def verify(model, pieces, rows, per_site=2, epsilons=(.25, .05, .01, .002), seed=0):
    """With every piece ON the sparse circuit IS the model, so the predicted first-order effect of
    removing one piece at one position must match a finite difference of the real forward pass.

    FP32 forwards, and pieces chosen among those with a measurable effect: at bf16 with a tiny epsilon
    the true change sits below the forward's own rounding noise, which tells you nothing.
    """
    import sparse_sketch
    circuit = sparse_sketch.Circuit(model, rows, pieces, probe_k=4, seed=seed, rules='exact', autocast=False)
    counts = [p.count for p in pieces]
    on = {n: torch.ones(circuit.batch, circuit.length, c, device='cuda') for n, c in zip(MODULES, counts)}
    effects = sparse_sketch.backward_effects(circuit, on)
    ids = rows[:, :-1]
    with torch.no_grad():
        base_logits = model(ids)[0].flatten(0, 1).float()
        base = torch.stack([F.cross_entropy(base_logits, circuit.labels[:, j], reduction='sum')
                            for j in range(circuit.probe_k)]).float()
    del base_logits
    results = []
    for site in MODULES:
        index = MODULES.index(site)
        rho = circuit.activation[site].view(circuit.batch, circuit.length, -1)
        size = (rho.abs() * torch.linalg.vector_norm(effects[site], dim=2))                # [B,T,C]
        flat = size.flatten()
        for choice in flat.topk(per_site).indices.tolist():                                # the loudest pieces
            b, rest = divmod(choice, circuit.length*size.shape[-1])
            t, piece = divmod(rest, size.shape[-1])
            value = float(rho[b, t, piece])
            write = pieces[index].write.float()[:, piece]
            for epsilon in epsilons:
                predicted = -epsilon * value * effects[site][b, t, :, piece]
                def perturb(_layer, _inputs, output, b=b, t=t, value=value, write=write, epsilon=epsilon):
                    changed = output.clone()
                    changed[b, t] = changed[b, t] - epsilon*value*write.to(changed.dtype)
                    return changed
                handle = model.get_submodule(site).register_forward_hook(perturb)
                try:
                    with torch.no_grad():
                        logits = model(ids)[0].flatten(0, 1).float()
                        actual = torch.stack([F.cross_entropy(logits, circuit.labels[:, j], reduction='sum')
                                              for j in range(circuit.probe_k)]).float() - base
                finally:
                    handle.remove()
                scale = max(float(predicted.norm()), float(actual.norm()), 1e-30)
                results.append(dict(site=site, piece=piece, position=t, epsilon=epsilon, rho=value,
                                    predicted=float(predicted.norm()), actual=float(actual.norm()),
                                    ratio=float(actual.norm()/max(float(predicted.norm()), 1e-30)),
                                    relative_error=float((predicted-actual).norm()/scale),
                                    cosine=float(F.cosine_similarity(predicted[None], actual[None])[0])))
                print(json.dumps(results[-1]), flush=True)
    for epsilon in epsilons:
        subset = [r for r in results if r['epsilon'] == epsilon and r['predicted'] > 1e-3]
        if not subset: continue
        print(json.dumps(dict(epsilon=epsilon, samples=len(subset),
                              worst_relative_error=max(r['relative_error'] for r in subset),
                              median_ratio=float(np.median([r['ratio'] for r in subset])),
                              min_cosine=min(r['cosine'] for r in subset))), flush=True)
    print('SKETCH_VERIFY_DONE', flush=True)
    return results


if __name__ == '__main__':
    main()
