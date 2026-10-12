"""Fisher sensor: K next tokens sampled from the model's own prediction, one cotangent each.

  score_c = |r_c.h| * sqrt(mean_k (g_k.u_c)^2),   g_k = d(-log p(y_k))/dy at the matrix output,
  y_k ~ p(.|context) from the intact model.

mean_k (g_k.u)^2 estimates u^T F u (the Fisher form at this token). Since KL(p || p_shifted)
is about ½ δ^T F δ for a small output change δ, this is the per-token, first-order score that
matches what KLKeep measures, with the "which label was correct" dependence averaged out.
Cached as the structured 'vjp' kind with K probes; output_count=K makes the norm an RMS.
Optional GIM rules (T=2 softmax VJP, Q/K/V scaling, frozen RMS) apply to every backward.
"""
import torch
import torch.nn.functional as F
from common import CONTEXT, MODULES


def capture_fisher(model, rows, k=8, microbatch=4, seed=0, gim=False, labels=False):
    """labels=True (tests only): use the true next tokens instead of samples, so k=1 equals capture()."""
    from gim_attribution import gim_context, ScaleGradient, RULES
    generator = torch.Generator(device='cuda').manual_seed(seed)
    scales = dict(q_proj=RULES['q_scale'], k_proj=RULES['k_scale'], v_proj=RULES['v_scale'])
    counts = dict(q_proj=0, k_proj=0, v_proj=0)
    entries = None
    with gim_context(model, temperature=RULES['softmax_temperature'] if gim else 1., freeze_rms=gim and RULES['freeze_rms']) as calls:
        for first in range(0, len(rows), microbatch):
            block = rows[first:first+microbatch]
            pre, post, hooks = {}, {}, []
            def retain(name):
                def hook(_m, inputs, out):
                    pre[name], post[name] = inputs[0], out
                    suffix = name.split('.')[-1]
                    if gim and suffix in scales:
                        counts[suffix] += 1
                        return ScaleGradient.apply(out, scales[suffix])
                return hook
            for name in MODULES: hooks.append(model.get_submodule(name).register_forward_hook(retain(name)))
            hooks.append(model.wte.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_()))
            try:
                with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(block[:, :-1])[0].flatten(0, 1)                       # [n, V]
                if labels:
                    samples = block[:, 1:].flatten()[:, None].expand(-1, k)
                else:
                    samples = torch.multinomial(logits.detach().float().softmax(-1), k, replacement=True, generator=generator)
                outputs = [post[n] for n in MODULES]
                for j in range(k):
                    # Same cross-entropy call as capture(), so labels=True, k=1 is bitwise the ordinary cache.
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = F.cross_entropy(logits, samples[:, j], reduction='sum')
                    grads = torch.autograd.grad(loss, outputs, retain_graph=j < k-1)
                    if entries is None:
                        entries = [dict(structured=True, kind='vjp', output_count=k, shared={'sensor': 'fisher_gim' if gim else 'fisher'},
                                        h=torch.empty(len(rows)*CONTEXT, pre[n].shape[-1], dtype=torch.bfloat16, device='cuda'),
                                        events={'cot': torch.empty(len(rows)*CONTEXT, k, g.shape[-1], dtype=torch.bfloat16, device='cuda')},
                                        net_rms=torch.zeros(len(rows)*CONTEXT, device='cuda')) for n, g in zip(MODULES, grads)]
                    lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
                    for n, g, e in zip(MODULES, grads, entries):
                        if j == 0: e['h'][lo:hi] = pre[n].detach().flatten(0, 1).bfloat16()
                        e['events']['cot'][lo:hi, j] = g.detach().flatten(0, 1).bfloat16()
                del logits, grads
            finally:
                for hook in hooks: hook.remove()
    expected = len(model.h)*((len(rows)+microbatch-1)//microbatch)
    if gim and (calls['softmax'] != expected or any(v != expected for v in counts.values())):
        raise RuntimeError(f'GIM rules did not cover all attention calls: {calls}, {counts}')
    return entries
