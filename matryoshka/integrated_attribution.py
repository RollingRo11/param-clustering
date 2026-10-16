"""Scalar-CE integrated gradients of component gates along a joint 0->1 path.

Only the 24 decomposed linear maps are scaled; embeddings, norms, biases,
and residual connections remain fixed. Five Gauss-Legendre nodes approximate
the signed integral. Magnitudes are taken AFTER integration.
"""
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from common import MODULES


def capture_scaled(model, rows, alpha, modules=MODULES, dtype=torch.bfloat16):
    inputs, outputs, hooks = {}, {}, []
    def capture(name):
        def hook(layer, args, output):
            inputs[name] = args[0]
            bias = layer.bias
            output = output*alpha if bias is None else (output-bias)*alpha+bias
            outputs[name] = output
            return output
        return hook
    try:
        for name in modules:
            hooks.append(model.get_submodule(name).register_forward_hook(capture(name)))
        hooks.append(model.wte.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_()))
        auto = torch.autocast(rows.device.type, dtype=dtype) if dtype else nullcontext()
        with torch.enable_grad(), auto:
            logits = model(rows[:, :-1])[0]
            loss = F.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten(), reduction='sum')
        gradients = torch.autograd.grad(loss, [outputs[n] for n in modules])
        return [(inputs[n].detach(), g.detach()) for n,g in zip(modules,gradients)], float(loss.detach())
    finally:
        for hook in hooks: hook.remove()


def signed_scores(cache, pieces, dtype=torch.bfloat16):
    auto = torch.autocast(pieces[0].read.device.type,dtype=dtype) if dtype else nullcontext()
    with torch.no_grad(), auto:
        return [((h @ p.read).float()*(g @ p.write).float()) if dtype else
                (h @ p.read)*(g @ p.write) for (h,g),p in zip(cache,pieces,strict=True)]


def integrated_scores(model, rows, pieces, k=5, modules=MODULES, dtype=torch.bfloat16):
    if k < 1: raise ValueError('Integration point count must be positive')
    nodes, weights = np.polynomial.legendre.leggauss(k)
    nodes, weights = (nodes+1)/2, weights/2
    integrated = None
    for alpha, weight in zip(nodes,weights):
        cache, _ = capture_scaled(model,rows,float(alpha),modules,dtype)
        values = signed_scores(cache,pieces,dtype)
        if integrated is None: integrated = [v*float(weight) for v in values]
        else:
            for total,value in zip(integrated,values): total.add_(value,alpha=float(weight))
        del cache, values
    # Signed completeness is diagnostic only, never a correction to the scores.
    attribution_sum = sum(float(v.double().sum()) for v in integrated)
    return [v.abs() for v in integrated], dict(signed_attribution_sum=attribution_sum,
            integration_points=k,nodes=nodes.tolist(),weights=weights.tolist(),
            path='all decomposed matrix gates jointly 0 -> 1; other parameters unchanged',
            quadrature='Gauss-Legendre; absolute value after signed integration')


def capture_integrated(model, rows, k=5, microbatch=4):
    """Token-major fixed caches consumed by the shared projected-Adam fitter."""
    nodes, weights = np.polynomial.legendre.leggauss(k)
    nodes, weights = (nodes+1)/2, weights/2
    n = len(rows)*(rows.shape[1]-1)
    entries = None
    for point,alpha in enumerate(nodes):
        for first in range(0,len(rows),microbatch):
            block=rows[first:first+microbatch]
            cache,_=capture_scaled(model,block,float(alpha))
            if entries is None:
                entries=[dict(structured=True,kind='scalar_ig',
                    h=torch.empty(n,k,h.shape[-1],device=rows.device,dtype=torch.bfloat16),
                    events={'g':torch.empty(n,k,g.shape[-1],device=rows.device,dtype=torch.bfloat16)},
                    shared={'quadrature_weights':weights.tolist(),'quadrature_nodes':nodes.tolist()},
                    net_rms=torch.zeros(n,device=rows.device)) for h,g in cache]
            lo=first*(rows.shape[1]-1); hi=lo+len(block)*(rows.shape[1]-1)
            for entry,(h,g) in zip(entries,cache):
                entry['h'][lo:hi,point].copy_(h.flatten(0,1))
                entry['events']['g'][lo:hi,point].copy_(g.flatten(0,1))
            del cache
    return entries
