"""Validate probes using the actual frozen 67M target, never a toy model."""
import json
import math
import torch
from common import MODULES, TARGET_SHA, batch, save_json
from gaussian_attribution import (trace, BOUNDARIES, fisher_probe, boundary_probes,
                                  capture_gaussian, calibrate_boundaries)


def validate(model, rows, initial, output):
    scales = calibrate_boundaries(model, rows[:4])
    report = dict(target_sha256=TARGET_SHA, contexts=1, context_length=512,
                  parameters=sum(p.numel() for p in model.parameters()), scales=scales,
                  derivative_checks=[], moment_checks=[])
    ids = rows[:1, :-1]
    with torch.no_grad():
        reference = model(ids)[0]
    with trace(model, boundaries=True) as (pre, post, hidden):
        logits = model(ids)[0]
        assert torch.equal(logits, reference), 'Capture changed intact forward'
        report['forward_bitwise_equal_fp32'] = True
    del reference, logits, pre, post, hidden

    # Three real paths: through attention, through a GELU, and through the final RMS/output head.
    for mi, source_pos in [(2, 255), (4, 383), (23, 511)]:
        name, piece = MODULES[mi], initial[mi]
        with trace(model, require_grad=False) as (pre, _, _):
            with torch.no_grad():
                model(ids)
            direction = torch.zeros(1, 512, piece.write.shape[0], device='cuda')
            # A rank-one component's local activation contribution.
            read = pre[name][0, source_pos].float() @ piece.read[:, 0]
            direction[0, source_pos] = read * piece.write[:, 0]
        del pre

        def measured(alpha):
            hook = model.get_submodule(name).register_forward_hook(lambda _m, _a, out: out+alpha*direction)
            try:
                with trace(model, boundaries=True, require_grad=False) as (_, _, h):
                    z = model(ids)[0]
                    return (z, *(h[n] for n in BOUNDARIES))
            finally:
                hook.remove()

        base, tangent = torch.autograd.functional.jvp(measured, torch.zeros((), device='cuda'),
                                                     torch.ones((), device='cuda'), strict=False)
        z, dz = base[0], tangent[0]
        pi = z.float().softmax(-1)
        exact_output = float(((pi*dz.square()).sum(-1)-(pi*dz).sum(-1).square()).sum())
        exact_boundary = sum(float(d.square().sum()) / scales[n]**2 for n, d in zip(BOUNDARIES, tangent[1:]))
        for coefficient in (0., .01):
            seed = 18273
            v = fisher_probe(z, torch.Generator(device='cuda').manual_seed(seed))
            signed = float((v*dz).sum())
            if coefficient:
                eta = boundary_probes(dict(zip(BOUNDARIES, base[1:])), scales, coefficient,
                                      torch.Generator(device='cuda').manual_seed(seed+2_000_000_011))
                signed += sum(float((e*d).sum()) for e, d in zip(eta, tangent[1:]))
            entries = capture_gaussian(model, rows[:1], seed=seed, coefficient=coefficient,
                                       scales=scales, fp32=True)
            actual = float((entries[mi]['events']['g'] * direction.flatten(0, 1)).sum())
            scale = math.sqrt(exact_output+coefficient*exact_boundary)
            rel = abs(actual-signed)/max(scale, 1e-12)
            assert rel < 2e-4, (name, coefficient, actual, signed, rel)
            report['derivative_checks'].append(dict(module=name, position=source_pos,
                coefficient=coefficient, vjp=actual, jvp_projection=signed,
                error_in_rms_units=rel, exact_metric_squared=scale**2))
            del entries, v
        # Moment check on actual logits/directions at selected affected prediction positions.
        # The VJP/JVP check above separately verifies propagation through the full 512-token graph.
        active = dz.square().sum(-1).flatten().topk(8).indices
        zz, dd = z.flatten(0, 1)[active], dz.flatten(0, 1)[active]
        pp = zz.softmax(-1)
        variance = float(((pp*dd.square()).sum(-1)-(pp*dd).sum(-1).square()).sum())
        draws = []
        generator = torch.Generator(device='cuda').manual_seed(9183)
        for _ in range(512):
            draws.append((fisher_probe(zz, generator)*dd).sum())
        draws = torch.stack(draws).double()/math.sqrt(max(variance, 1e-30))
        for p in (.5, .641, 1.):
            kappa = 2**(p/2)*math.gamma((p+1)/2)/math.sqrt(math.pi)
            values = draws.abs().pow(p)/kappa
            mean, se = float(values.mean()), float(values.std()/math.sqrt(len(values)))
            assert abs(mean-1) < max(5*se, .06), (name, p, mean, se)
            report['moment_checks'].append(dict(module=name, p=p, normalized_mean=mean, standard_error=se, probes=512))
        del base, tangent, z, dz, pi, direction, zz, dd, pp, draws
    # Final label is not an input token: changing it must not change any cache tensor.
    a = capture_gaussian(model, rows[:1], seed=13, coefficient=.01, scales=scales)
    changed = rows[:1].clone(); changed[:, -1] = (changed[:, -1]+1) % model.config.vocab_size
    b = capture_gaussian(model, changed, seed=13, coefficient=.01, scales=scales)
    assert all(torch.equal(x['h'], y['h']) and torch.equal(x['events']['g'], y['events']['g']) for x,y in zip(a,b))
    assert all(not v.requires_grad for e in a for v in [e['h'], e['events']['g']])
    report['label_independent_and_deterministic'] = True
    report['cached_tensors_detached'] = True
    report['passed'] = True
    save_json(report, output)
    print(json.dumps(dict(validation=report)), flush=True)
    return scales
