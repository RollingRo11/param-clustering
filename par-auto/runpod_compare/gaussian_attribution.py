"""One intact-model VJP: Gaussian categorical-Fisher and optional residual probes.

The unsmoothed absolute p-moment is kappa_p times the full metric norm^p.
The existing smoothed concentration loss is retained for a controlled trial.
No factors, component selection, labels, or model perturbations enter capture.
"""
from contextlib import contextmanager
import math
import torch
from common import MODULES, CONTEXT


BOUNDARIES = tuple(f'h.{i}.{kind}' for i in range(4) for kind in ('post_attn', 'post_mlp'))


@contextmanager
def trace(model, *, boundaries=False, require_grad=True):
    pre, post, hidden, hooks = {}, {}, {}, []
    try:
        for name in MODULES:
            def grab(_m, inputs, out, name=name):
                pre[name], post[name] = inputs[0], out
            hooks.append(model.get_submodule(name).register_forward_hook(grab))
        if require_grad:
            hooks.append(model.wte.register_forward_hook(lambda _m, _a, out: out.detach().requires_grad_()))
        if boundaries:
            for i, block in enumerate(model.h):
                def before_mlp(_m, inputs, i=i):
                    hidden[f'h.{i}.post_attn'] = inputs[0]
                def after_mlp(_m, _a, out, i=i):
                    hidden[f'h.{i}.post_mlp'] = out
                hooks.append(block.rms_2.register_forward_pre_hook(before_mlp))
                hooks.append(block.register_forward_hook(after_mlp))
        yield pre, post, hidden
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def fisher_probe(logits, generator, chunk_positions=128):
    """Cov(v_t)=diag(pi_t)-pi_t*pi_t^T; independent Gaussian seeds per output position."""
    flat = logits.detach().flatten(0, -2)
    result = torch.empty_like(flat)
    for start in range(0, len(flat), chunk_positions):
        pi = flat[start:start+chunk_positions].float().softmax(-1)
        noise = torch.randn(pi.shape, device=pi.device, dtype=torch.float32, generator=generator)
        noise.mul_(pi.sqrt())
        noise.sub_(pi * noise.sum(-1, keepdim=True))
        result[start:start+len(pi)] = noise.to(result.dtype)
    return result.reshape_as(logits)


@torch.no_grad()
def boundary_probes(hidden, scales, coefficient, generator):
    return [torch.randn(hidden[n].shape, device=hidden[n].device, dtype=torch.float32,
                        generator=generator).mul_(math.sqrt(coefficient) / scales[n]).to(hidden[n].dtype)
            for n in BOUNDARIES]


@torch.no_grad()
def calibrate_boundaries(model, rows, microbatch=4):
    sums, count = dict.fromkeys(BOUNDARIES, 0.), 0
    for first in range(0, len(rows), microbatch):
        with trace(model, boundaries=True, require_grad=False) as (_, _, hidden):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(rows[first:first+microbatch, :-1])
            # RMS vector norm: E||delta b||^2 / E||b||^2, rather than width times that metric.
            for n in BOUNDARIES:
                sums[n] += float(hidden[n].float().square().sum())
            count += hidden[BOUNDARIES[0]].shape[0] * hidden[BOUNDARIES[0]].shape[1]
    return {n: math.sqrt(max(v/count, 1e-12)) for n, v in sums.items()}


def capture_gaussian(model, rows, *, seed, coefficient=0., scales=None, microbatch=4,
                     fp32=False):
    if coefficient < 0 or (coefficient and (scales is None or set(scales) != set(BOUNDARIES))):
        raise ValueError('Positive boundary coefficient requires fixed calibrated scales')
    if rows.shape[1] != CONTEXT+1:
        raise ValueError('Expected the existing 512-position contexts plus last label')
    entries = None
    for first in range(0, len(rows), microbatch):
        block = rows[first:first+microbatch]
        gen = torch.Generator(device='cuda').manual_seed(seed+first)
        boundary_gen = torch.Generator(device='cuda').manual_seed(seed+first+2_000_000_011)
        with trace(model, boundaries=coefficient > 0) as (pre, post, hidden):
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not fp32):
                logits = model(block[:, :-1])[0]
            outs, probes = [logits], [fisher_probe(logits, gen)]
            if coefficient:
                outs += [hidden[n] for n in BOUNDARIES]
                probes += boundary_probes(hidden, scales, coefficient, boundary_gen)
            grads = torch.autograd.grad(outs, [post[n] for n in MODULES], grad_outputs=probes)
            dtype = torch.float32 if fp32 else torch.bfloat16
            if entries is None:
                entries = [dict(structured=True, kind='scalar_loss',
                                shared={'sensor': 'gaussian_fisher', 'coefficient': coefficient},
                                h=torch.empty(len(rows)*CONTEXT, pre[n].shape[-1], dtype=dtype, device='cuda'),
                                events={'g': torch.empty(len(rows)*CONTEXT, g.shape[-1], dtype=dtype, device='cuda')},
                                net_rms=torch.zeros(len(rows)*CONTEXT, device='cuda'))
                           for n, g in zip(MODULES, grads)]
            lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
            for n, g, e in zip(MODULES, grads, entries):
                e['h'][lo:hi] = pre[n].detach().flatten(0, 1).to(dtype)
                e['events']['g'][lo:hi] = g.detach().flatten(0, 1).to(dtype)
            del logits, grads, outs, probes
    return entries
