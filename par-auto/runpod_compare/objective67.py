"""Screen objectives against the exact VPD witness; frozen 67M, Explorer.

24 intact-trace objectives, 12 subset-conditioned diagnostics, and projected
one-step descent probes of all 24 intact-trace objectives. No sparse target
forward or CI enters factor-training losses. See objective67.md.
"""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
import time

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/ifr67_20260915'))
import ifr67 as flow
bridge, audit = flow.bridge_tools, flow.audit
import numpy as np
import torch
import torch.nn.functional as F
from common import MODULES, TARGET_SHA, target, tokens, save_json, sha, upload, credentials
from train_pair import core, capture as capture_true
from gim_attribution import capture_gim
from fisher_attribution import capture_fisher

EXPERIMENT = 'objective67_20260915'
OUT = ROOT/'output'/EXPERIMENT
SENSORS = ['gim', 'gim_true_norm', 'true_grad', 'write', 'consumer_tangent', 'consumer_finite_rms']
POWERS = [.25, .5, .641]
OBJECTIVES = [(sensor, power) for sensor in SENSORS for power in POWERS+['count']]
STEPS = [.0001, .001, .003]
TRAIN_ROWS = [10000, 10001]
DISCOVERY = list(range(448, 452))
CONFIRMATION = list(range(452, 456))
CONDITIONAL = ['local_mse', 'local_l1', 'local_cosine', 'local_whitened', 'consumer_mse',
               'sublayer_mse', 'sublayer_rms', 'block_mse', 'block_cosine', 'block_rms',
               'block_rms_plus_local_01', 'block_rms_plus_local_05']


def key(sensor, power):
    return f'{sensor}_{power}'


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter()-start


def read_pieces(path):
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['target_sha256'] == TARGET_SHA
    return [core.FreePieces(p['read'].cuda().float(), p['write'].cuda().float()) for p in saved['factors']]


@torch.no_grad()
def trace(model, rows, sensors=True):
    final = {}
    hook = model.ln_f.register_forward_pre_hook(lambda _m, args: final.update(h=args[0][0].detach()))
    try:
        (base, inputs, outputs, attention, checks), elapsed = timed(lambda: flow.capture(model, rows[:, :-1]))
    finally:
        hook.remove()
    inputs['ln_f'] = final['h']
    ctx = dict(rows=rows, base=base.detach(), inputs=inputs, outputs=outputs, attention=attention,
               checks=checks, timing={'trace': elapsed}, cache={})
    if sensors:
        for name, fn in [('gim', lambda: capture_gim(model, rows, microbatch=1)),
                         ('gim_true_norm', lambda: capture_gim(model, rows, microbatch=1, freeze_rms=False)),
                         ('true_grad', lambda: capture_true(model, rows, microbatch=1))]:
            ctx['cache'][name], ctx['timing'][name] = timed(fn)
    return ctx


def next_norm(model, ctx, i):
    layer, kind = i//6, i%6
    if kind == 3:
        name = f'h.{layer}.rms_2'
    else:
        name = f'h.{layer+1}.rms_1' if layer < 3 else 'ln_f'
    return model.get_submodule(name), ctx['inputs'][name]


def positive_sqrt(x):
    return x.clamp_min(1e-16).sqrt()


def scores(model, ctx, pieces, sensor):
    values = []
    for i, (name, p) in enumerate(zip(MODULES, pieces, strict=True)):
        if sensor in ctx['cache']:
            e = ctx['cache'][sensor][i]
            a = e['h'].float() @ p.read
            values.append(a.abs()*(e['events']['g'].float() @ p.write).abs())
            continue
        h = ctx['inputs'][name]
        a = h @ p.read
        norm2 = p.write.square().sum(0)
        if sensor == 'write' or i%6 in [0, 1, 2]:
            values.append(a.abs()*positive_sqrt(norm2))
        elif i%6 == 4:
            x = ctx['outputs'][name]
            t = math.sqrt(2/math.pi)*(x+.044715*x.pow(3))
            th = t.tanh()
            derivative = .5*(1+th)+.5*x*math.sqrt(2/math.pi)*(1+3*.044715*x.square())*(1-th.square())
            values.append(a.abs()*positive_sqrt(derivative.square() @ p.write.square()))
        else:
            norm, x = next_norm(model, ctx, i)
            gamma = norm.weight.float()
            d = x.shape[-1]
            s0sq = x.square().mean(-1, keepdim=True)+norm.variance_epsilon
            dot = x @ p.write
            weighted_dot = (x*gamma.square()) @ p.write
            weighted_u2 = (gamma[:, None]*p.write).square().sum(0)
            weighted_x2 = (x*gamma).square().sum(-1, keepdim=True)
            if sensor == 'consumer_tangent':
                c = dot/(d*s0sq)
                square = (weighted_u2-2*c*weighted_dot+c.square()*weighted_x2)/s0sq
                values.append(a.abs()*positive_sqrt(square))
            else:
                # Exact finite RMS change for x -> x-a*u, using two TC dot
                # products; no T*C*D tensor or per-component model execution.
                change = (-2*a*dot+a.square()*norm2)/d
                s0 = s0sq.sqrt()
                s1 = (s0sq+change).clamp_min(norm.variance_epsilon).sqrt()
                inv_difference = change/(s0*s1*(s0+s1))
                c = a/s1
                square = inv_difference.square()*weighted_x2+2*inv_difference*c*weighted_dot+c.square()*weighted_u2
                values.append(positive_sqrt(square))
    return values


def reduce_scores(values, sensor, power, scales):
    per_module = []
    for i, value in enumerate(values):
        if power == 'count':
            tau = scales['tau'][sensor][i]
            scalar = (value/(value+tau)).sum(-1).mean()
        else:
            scalar = ((value+1e-6).pow(power)-1e-6**power).sum(-1).mean()
        per_module.append(scalar)
    vector = torch.stack(per_module)
    normalizer = torch.tensor(scales['loss'][key(sensor, power)], device='cuda')
    return (vector/normalizer).mean()


@torch.no_grad()
def calibrate(model, contexts, pieces):
    scales = dict(tau={}, loss={})
    for sensor in SENSORS:
        arrays = [scores(model, ctx, pieces, sensor) for ctx in contexts]
        scales['tau'][sensor] = [max(1e-12, .01*st.mean(float(x[i].mean()) for x in arrays)) for i in range(24)]
        for power in POWERS+['count']:
            if power == 'count':
                values = [st.mean(float((x[i]/(x[i]+scales['tau'][sensor][i])).sum(-1).mean()) for x in arrays) for i in range(24)]
            else:
                values = [st.mean(float(((x[i]+1e-6).pow(power)-1e-6**power).sum(-1).mean()) for x in arrays) for i in range(24)]
            scales['loss'][key(sensor, power)] = [max(1e-12, v) for v in values]
    return scales


def rel_mse(pred, ref):
    denom = ref.square().sum(-1)
    return ((pred-ref).square().sum(-1)/denom.clamp_min(1e-4*denom.mean()+1e-12)).mean()


def cosine(pred, ref):
    return (1-F.cosine_similarity(pred, ref, dim=-1, eps=1e-8)).clamp_min(0).mean()


def synth(h, p, mask):
    return ((h @ p.read)*mask) @ p.write.T


def attention_mix(model, layer, q, k, v):
    T = len(q)
    att = model.h[layer].attn
    q = q.reshape(1, T, 6, 128).transpose(1, 2)
    k = k.reshape(1, T, 6, 128).transpose(1, 2)
    q, k = att.apply_rotary_pos_emb(q, k, att.rotary_cos[:T][None], att.rotary_sin[:T][None])
    z = (q @ k.transpose(-1, -2))/math.sqrt(128)
    z = z.masked_fill(~torch.ones(T, T, device='cuda', dtype=torch.bool).tril(), -float('inf'))
    a = z.softmax(-1)
    return (a @ v.reshape(1, T, 6, 128).transpose(1, 2)).transpose(1, 2).reshape(T, 768)


def conditional_losses(model, ctx, pieces, keep):
    parts = keep.split([p.count for p in pieces], -1)
    selected = [synth(ctx['inputs'][n], p, m) for n, p, m in zip(MODULES, pieces, parts, strict=True)]
    acc = {k: [] for k in CONDITIONAL if '_plus_' not in k}
    for i, (name, pred) in enumerate(zip(MODULES, selected, strict=True)):
        ref = ctx['outputs'][name]
        error = pred-ref
        acc['local_mse'].append(rel_mse(pred, ref))
        denom = ref.abs().sum(-1)
        acc['local_l1'].append((error.abs().sum(-1)/denom.clamp_min(1e-4*denom.mean()+1e-12)).mean())
        acc['local_cosine'].append(cosine(pred, ref))
        scale = ref.square().mean(0)
        acc['local_whitened'].append((error.square()/scale.clamp_min(.01*scale.mean()+1e-12)).mean())
        if i%6 in [3, 5]:
            norm, residual = next_norm(model, ctx, i)
            acc['consumer_mse'].append(rel_mse(norm(residual+error), norm(residual)))
        elif i%6 == 4:
            acc['consumer_mse'].append(rel_mse(F.gelu(pred, approximate='tanh'), F.gelu(ref, approximate='tanh')))
        else:
            acc['consumer_mse'].append(rel_mse(pred, ref))
    for layer in range(4):
        i = 6*layer
        h0 = ctx['inputs'][f'h.{layer}.rms_1']
        clean_h1 = ctx['inputs'][f'h.{layer}.rms_2']
        next_name = f'h.{layer+1}.rms_1' if layer < 3 else 'ln_f'
        clean_h2 = ctx['inputs'][next_name]
        next_layer_norm = model.get_submodule(next_name)
        att_input = attention_mix(model, layer, *selected[i:i+3])
        sparse_o = synth(att_input, pieces[i+3], parts[i+3])
        h1 = h0+sparse_o
        # MLP evaluated on the original MLP input: sublayer-local reference.
        independent_down = synth(F.gelu(selected[i+4], approximate='tanh'), pieces[i+5], parts[i+5])
        sub_h2 = clean_h1+independent_down
        acc['sublayer_mse'].extend([rel_mse(h1, clean_h1), rel_mse(sub_h2, clean_h2)])
        acc['sublayer_rms'].extend([rel_mse(model.h[layer].rms_2(h1), model.h[layer].rms_2(clean_h1)),
                                    rel_mse(next_layer_norm(sub_h2), next_layer_norm(clean_h2))])
        # Whole selected block, still started independently from intact h0.
        fc = synth(model.h[layer].rms_2(h1), pieces[i+4], parts[i+4])
        down = synth(F.gelu(fc, approximate='tanh'), pieces[i+5], parts[i+5])
        h2 = h1+down
        acc['block_mse'].append(rel_mse(h2, clean_h2))
        acc['block_cosine'].append(cosine(h2, clean_h2))
        acc['block_rms'].append(rel_mse(next_layer_norm(h2), next_layer_norm(clean_h2)))
    result = {k: torch.stack(v).mean() for k, v in acc.items()}
    result['block_rms_plus_local_01'] = result['block_rms']+.1*result['consumer_mse']
    result['block_rms_plus_local_05'] = result['block_rms']+.5*result['consumer_mse']
    return result


@torch.no_grad()
def native_support(trainer, ctx):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        cached = trainer.model(ctx['rows'][:, :-1], cache_type='input')
        ci = trainer.model.calc_causal_importances(cached.cache, sampling='continuous').lower_leaky
        return torch.cat([ci[n][0].float() for n in MODULES], -1)


@torch.no_grad()
def endpoint(model, ctx, pieces, keep):
    return float(audit.loss(audit.run(model, ctx['rows'][:, :-1], pieces, keep.float()), ctx['base'].log_softmax(-1)))


@torch.no_grad()
def validate(model, ctx, pieces):
    checks = {}
    all_kept = torch.ones(512, sum(p.count for p in pieces), device='cuda')
    losses = conditional_losses(model, ctx, pieces, all_kept)
    checks['all_kept_losses'] = {k: float(v) for k, v in losses.items()}
    assert max(checks['all_kept_losses'].values()) < 1e-4, checks
    for i in [3, 5, 21, 23]:
        norm, x = next_norm(model, ctx, i)
        p = pieces[i]
        a = (ctx['inputs'][MODULES[i]][:3] @ p.read[:, :7])
        z = a[..., None]*p.write[:, :7].T[None]
        explicit = (norm(x[:3])[:, None]-norm(x[:3, None]-z)).norm(dim=-1)
        calculated = scores(model, ctx, pieces, 'consumer_finite_rms')[i][:3, :7]
        torch.testing.assert_close(calculated, explicit, rtol=.003, atol=3e-5)
        checks[f'finite_rms_{i}_max_error'] = float((explicit-calculated).abs().max())
    # Zero gates should not look as faithful as all-kept under raw/finite losses.
    zero = conditional_losses(model, ctx, pieces, torch.zeros_like(all_kept))
    assert float(zero['block_rms']) > float(losses['block_rms'])+1e-4
    checks['all_removed_block_rms'] = float(zero['block_rms'])
    return checks


@torch.no_grad()
def scan_context(model, ctx, states, scales, old, trainer, old5, trainer5, sequence):
    native = native_support(trainer, ctx)>0
    inherited = bridge.remap(native, old, states['vpd_exact'])
    counts = native.sum(-1)
    own5 = bridge.remap(native_support(trainer5, ctx), old5, states['vpd5k_exact'])
    with torch.enable_grad():
        fisher, fisher_seconds = timed(lambda: capture_fisher(model, ctx['rows'], k=64, seed=6419000+sequence, microbatch=1))
    record = dict(sequence=sequence, split='discovery' if sequence in DISCOVERY else 'confirmation',
                  count_mean=float(counts.float().mean()), timing=ctx['timing'], fisher_seconds=fisher_seconds, states={})
    for name, pieces in states.items():
        entry = dict(maskfree={}, scoring_seconds={}, variants={})
        write_scores = None
        for sensor in SENSORS:
            values, seconds = timed(lambda: scores(model, ctx, pieces, sensor))
            entry['scoring_seconds'][sensor] = seconds
            for power in POWERS+['count']:
                entry['maskfree'][key(sensor, power)] = float(reduce_scores(values, sensor, power, scales))
            if sensor == 'write':
                write_scores = torch.cat(values, -1)
        fs, _ = audit.rank_and_census(fisher, ctx['cache']['gim'], pieces)
        variants = dict(fisher=bridge.variable_support(fs, counts), write=bridge.variable_support(write_scores, counts))
        if name == 'vpd_exact' or name.startswith('descent_'):
            variants['inherited_native'] = inherited
        if name == 'vpd5k_exact':
            variants['native_ci_matched_budget'] = bridge.variable_support(own5, counts)
        generator = torch.Generator(device='cuda').manual_seed(73000+sequence)
        random_scores = torch.rand(inherited.shape, generator=generator, device='cuda')
        variants['random_native_allocation'] = torch.cat([
            bridge.variable_support(v, m.sum(-1)) for v, m in zip(random_scores.split([p.count for p in pieces], -1),
                                                                inherited.split([p.count for p in pieces], -1), strict=True)], -1)
        for label, keep in variants.items():
            assert torch.equal(keep.sum(-1), counts)
            values, elapsed = timed(lambda: conditional_losses(model, ctx, pieces, keep.float()))
            entry['variants'][label] = dict(kl=endpoint(model, ctx, pieces, keep),
                                            conditional={k: float(v) for k, v in values.items()},
                                            conditional_suite_seconds=elapsed)
        record['states'][name] = entry
    return record, inherited


def clone(pieces, grad=False):
    return [core.FreePieces(p.read.detach().clone().requires_grad_(grad), p.write.detach().clone().requires_grad_(grad)) for p in pieces]


def probe(model, train_ctx, heldout, initial, scales, sensor, power):
    pieces = clone(initial, grad=True)
    params = [v for p in pieces for v in [p.read, p.write]]
    def backward():
        losses = []
        for ctx in train_ctx:
            loss = reduce_scores(scores(model, ctx, pieces, sensor), sensor, power, scales)
            (loss/len(train_ctx)).backward()
            losses.append(float(loss.detach()))
        return st.mean(losses)
    before, elapsed = timed(backward)
    grads = [p.grad.detach().clone() for p in params]
    assert all(torch.isfinite(g).all() for g in grads)
    gnorm = torch.stack([g.double().square().sum() for g in grads]).sum().sqrt()
    pnorm = torch.stack([p.detach().double().square().sum() for p in params]).sum().sqrt()
    result = dict(objective=key(sensor, power), before=before, factor_forward_backward_seconds=elapsed/len(train_ctx),
                  cache_seconds=st.mean(c['timing'].get(sensor, c['timing']['trace']) for c in train_ctx),
                  train_rows=TRAIN_ROWS, steps=[])
    for size in STEPS:
        with torch.no_grad():
            proposed = []
            for i, p in enumerate(initial):
                proposed.append(core.FreePieces(p.read-size*float(pnorm/gnorm)*grads[2*i],
                                                 p.write-size*float(pnorm/gnorm)*grads[2*i+1]))
            candidate, projection_seconds = timed(lambda: [core.projected_pieces(model.get_submodule(n).weight, p)
                                                            for n, p in zip(MODULES, proposed, strict=True)])
            errors = [core.reconstruction_error(model.get_submodule(n).weight, p) for n, p in zip(MODULES, candidate, strict=True)]
            assert max(errors) < 1e-5
            after = st.mean(float(reduce_scores(scores(model, c, candidate, sensor), sensor, power, scales)) for c in train_ctx)
            displacement = math.sqrt(sum(float((q.read-p.read).double().square().sum()+(q.write-p.write).double().square().sum())
                                         for q, p in zip(candidate, initial, strict=True)))/float(pnorm)
            kls = [endpoint(model, c, candidate, c['native']) for c in heldout]
            result['steps'].append(dict(requested_relative_step=size, projected_relative_displacement=displacement,
                                        after=after, objective_decreased=after<before, heldout_kl=kls,
                                        heldout_kl_mean=st.mean(kls), projection_seconds=projection_seconds,
                                        max_matrix_error=max(errors)))
    # Choose step from TRAIN objective only. Held-out KL never selects the step.
    acceptable = [s for s in result['steps'] if s['objective_decreased']]
    result['selected'] = min(acceptable, key=lambda x: x['after']) if acceptable else None
    return result


def summarize(plan, records, probes):
    summary = dict(plan=plan, maskfree={}, conditional={}, records=records, probes=probes)
    for sensor, power in OBJECTIVES:
        k = key(sensor, power)
        result = {}
        for split in ['discovery', 'confirmation']:
            rows = [r for r in records if r['split'] == split]
            get = lambda r, n: r['states'][n]['maskfree'][k]
            vpd = [get(r, 'vpd_exact') for r in rows]
            gim = [get(r, 'gim5k') for r in rows]
            trend = [sum(get(r, b)>get(r, a) for a, b in zip(['vpd_exact', 'descent_20', 'descent_100'],
                                                               ['descent_20', 'descent_100', 'descent_200']))/3 for r in rows]
            result[split] = dict(vpd_mean=st.mean(vpd), gim_mean=st.mean(gim), ratio=st.mean(vpd)/st.mean(gim),
                                 vpd_preferred_fraction=st.mean(float(a<b) for a, b in zip(vpd, gim)),
                                 worsening_trajectory_increases_loss_fraction=st.mean(trend))
        summary['maskfree'][k] = result
    for k in CONDITIONAL:
        result = {}
        for split in ['discovery', 'confirmation']:
            rows = [r for r in records if r['split'] == split]
            get = lambda r, n, s: r['states'][n]['variants'][s]['conditional'][k]
            vpd = [get(r, 'vpd_exact', 'inherited_native') for r in rows]
            gim = [get(r, 'gim5k', 'fisher') for r in rows]
            result[split] = dict(vpd_mean=st.mean(vpd), gim_mean=st.mean(gim), ratio=st.mean(vpd)/st.mean(gim),
                                 vpd_preferred_fraction=st.mean(float(a<b) for a, b in zip(vpd, gim)),
                                 native_beats_fisher_fraction=st.mean(float(get(r,'vpd_exact','inherited_native')<get(r,'vpd_exact','fisher')) for r in rows),
                                 native_beats_random_fraction=st.mean(float(get(r,'vpd_exact','inherited_native')<get(r,'vpd_exact','random_native_allocation')) for r in rows),
                                 worsening_trajectory_increases_loss_fraction=st.mean(sum(get(r,b,'inherited_native')>get(r,a,'inherited_native') for a,b in zip(['vpd_exact','descent_20','descent_100'],['descent_20','descent_100','descent_200']))/3 for r in rows))
        summary['conditional'][k] = result
    summary['baseline_heldout_kl'] = st.mean(r['states']['vpd_exact']['variants']['inherited_native']['kl'] for r in records if r['split']=='confirmation')
    save_json(summary, OUT/'comparison.json')
    lines = ['# Objective screening against the exact VPD witness', '',
             'Frozen 67M target on Explorer. 24 intact-trace objectives and 12 subset-conditioned diagnostics. '
             'Calibration and descent gradients use two training contexts; discovery and confirmation each use four separate validation contexts. '
             'All comparisons share exact-sum capacity and the published native per-position count. This is a diagnostic screen, not a learned sparse method.', '',
             '## Intact-trace objectives', '',
             '| Objective | Discovery VPD/GIM loss | Confirmation VPD/GIM loss | Worsening trajectory increases loss | Selected one-step held-out KL |',
             '|---|---:|---:|---:|---:|']
    for k, values in sorted(summary['maskfree'].items(), key=lambda x:x[1]['discovery']['ratio']):
        probe_result = probes.get(k, {}).get('selected')
        kl = f"{probe_result['heldout_kl_mean']:.5f}" if probe_result else 'no decreasing step'
        lines.append(f"| {k} | {values['discovery']['ratio']:.4f} | {values['confirmation']['ratio']:.4f} | {values['confirmation']['worsening_trajectory_increases_loss_fraction']:.2f} | {kl} |")
    lines += ['', f"Unmodified VPD witness confirmation KL: **{summary['baseline_heldout_kl']:.5f}**. "
              'A ratio below one favors VPD. The trajectory statistic should be high when a loss objects to the known GIM-induced damage. '
              'Descent steps are selected by training objective only, never by confirmation KL. All step sizes and per-context outcomes remain in comparison.json.', '',
              '## Subset-conditioned diagnostics', '',
              '| Objective | Discovery VPD/GIM | Confirmation VPD/GIM | Native beats Fisher | Native beats random | Worsening trajectory increases loss |',
              '|---|---:|---:|---:|---:|---:|']
    for k, values in sorted(summary['conditional'].items(), key=lambda x:x[1]['discovery']['ratio']):
        d,c = values['discovery'], values['confirmation']
        lines.append(f"| {k} | {d['ratio']:.4f} | {c['ratio']:.4f} | {c['native_beats_fisher_fraction']:.2f} | {c['native_beats_random_fraction']:.2f} | {c['worsening_trajectory_increases_loss_fraction']:.2f} |")
    lines += ['', 'Conditional diagnostics receive a fixed candidate support. They are not strictly mask-free training objectives. '
              'No factor updates used them. Full selected-model KL is evaluation only. '
              'The twelve diagnostics share intermediate calculations; suite timing must not be reported as the isolated cost of any one objective.', '',
              '## Limits', '',
              'The favorable VPD support is inherited from its trained CI network. Acceptance is a necessary diagnostic gate, not a way to discover that support. '
              'A single normalized projected gradient step is not a 5k-step training result or an Adam trajectory. '
              'Descent uses the existing conditional exact-sum projection, which is not an orthogonal joint-manifold projection. '
              'The raw gradient is normalized in the existing factor coordinates; actual postprojection displacement is reported. '
              'Whole training cost at the production batch size remains unmeasured. No claim of a lower L0 floor follows from this screen.', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    return summary


def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed(); OUT.mkdir(parents=True, exist_ok=True)
    model = target()
    training, manifest = tokens('train')
    validation, _ = tokens('val')
    ours, _, oursmeta = audit.shared.load(model, 'gim_5000')
    late, _, latemeta = audit.shared.load(model, 'gim_7783')
    old, trainer, oldmeta = audit.shared.load(model, 'vpd_published')
    old5, trainer5, old5meta = audit.shared.load(model, 'vpd_5000')
    audit.shared.SPECS['gim1b'] = ('parauto', 30518, 'checkpoints/published_vs_finished/parauto.pth')
    big, _, bigmeta = audit.shared.load(model, 'gim1b')
    exact = read_pieces(ROOT/'output/gap67_bridge_20260915/bridge_model.pth')
    exact5, conversion = bridge.convert(model, old5, ours)
    states = dict(vpd_exact=exact, gim5k=ours, gim7783=late, gim1b_original_ce=big, vpd5k_exact=exact5)
    for step in [20,100,200]:
        states[f'descent_{step}'] = read_pieces(ROOT/f'output/gap67_descent_20260915/{step}.pth')
    train_ctx = [trace(model, torch.tensor(np.array(training[r:r+1], dtype=np.int64), device='cuda')) for r in TRAIN_ROWS]
    scales = calibrate(model, train_ctx, ours)
    save_json(scales, OUT/'normalizers.json')
    save_json(validate(model, train_ctx[0], exact), OUT/'validation.json')
    plan = dict(source_sha256=sha(__file__), target_sha256=TARGET_SHA, data=manifest, train_rows=TRAIN_ROWS,
                discovery=DISCOVERY, confirmation=CONFIRMATION, maskfree_objectives=[key(*x) for x in OBJECTIVES],
                conditional_objectives=CONDITIONAL, probe_relative_steps=STEPS, sources=dict(gim5k=oursmeta,gim7783=latemeta,
                vpd=oldmeta,vpd5k=old5meta,gim1b_original_ce=bigmeta), vpd5k_conversion=conversion,
                bridge_sha256=sha(ROOT/'output/gap67_bridge_20260915/bridge_model.pth'),
                trajectory_sha256={str(s):sha(ROOT/f'output/gap67_descent_20260915/{s}.pth') for s in [20,100,200]},
                note='Subset-conditioned losses are diagnostics only. Factor gradient probes use intact traces only. '
                     'No Gaussian probes, sparse target forwards, or CI in training losses. Fisher is an evaluation selector only.')
    save_json(plan, OUT/'plan.json'); print('OBJECTIVE67_PREFLIGHT_PASSED', flush=True)
    records, heldout = [], []
    for sequence in DISCOVERY+CONFIRMATION:
        ctx = trace(model, torch.tensor(np.array(validation[sequence:sequence+1], dtype=np.int64), device='cuda'))
        record, native = scan_context(model, ctx, states, scales, old, trainer, old5, trainer5, sequence)
        save_json(record, OUT/f'{sequence}.json'); records.append(record)
        if sequence in CONFIRMATION:
            heldout.append(dict(rows=ctx['rows'], base=ctx['base'], native=native))
        print(json.dumps(dict(scored_sequence=sequence, native_kl=record['states']['vpd_exact']['variants']['inherited_native']['kl'],
                              gim_kl=record['states']['gim5k']['variants']['fisher']['kl'])), flush=True)
        del ctx
    summarize(plan, records, {})
    probes = {}
    for sensor, power in OBJECTIVES:
        result = probe(model, train_ctx, heldout, exact, scales, sensor, power)
        probes[key(sensor,power)] = result
        save_json(result, OUT/f'probe_{key(sensor,power)}.json')
        selected = result['selected']
        print(json.dumps(dict(probe=key(sensor,power), before=result['before'],
                              after=selected['after'] if selected else None,
                              heldout_kl=selected['heldout_kl_mean'] if selected else None)), flush=True)
    summarize(plan, records, probes)
    receipts = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in ['complete.json', 'r2_receipts.json']:
            receipts.append(upload(p, f'experiments/{EXPERIMENT}/{p.name}'))
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.is_file(): receipts.append(upload(p, f'experiments/{EXPERIMENT}/source/{p.name}'))
    save_json(receipts, OUT/'r2_receipts.json')
    save_json(dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(OUT/'comparison.json')),OUT/'complete.json')
    upload(OUT/'complete.json', f'experiments/{EXPERIMENT}/complete.json')
    print('OBJECTIVE67_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
