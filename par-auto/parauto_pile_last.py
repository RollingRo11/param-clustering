"""Exact final-MLP logit-effect norms, compressed through the full vocabulary head.

No probes, eigenvalue truncation or detached normalization derivatives. Scores
are first-order effects at the frozen model's activations; finite ablations use
the complete nonlinear final MLP, final RMSNorm and vocabulary head.
"""
import math
import torch
import torch.nn.functional as F
import parauto_core as core

MODULES = ('h.3.mlp.c_fc', 'h.3.mlp.down_proj')


def gelu_derivative(z):
    k = math.sqrt(2 / math.pi)
    t = torch.tanh(k * (z + .044715 * z.pow(3)))
    return .5 * (1 + t) + .5 * z * (1 - t.square()) * k * (1 + 3 * .044715 * z.square())


@torch.no_grad()
def head_factor(model, context):
    # R^T R = diag(gamma) W_U^T W_U diag(gamma)/(T V).
    # Cholesky retains all 768 directions; no ridge or spectral truncation.
    head = model.lm_head.weight.double() * model.ln_f.weight.double()[None, :]
    gram = head.T @ head / (context * head.shape[0])
    r = torch.linalg.cholesky(gram).T
    error = float((r.T @ r - gram).norm() / gram.norm())
    return r.float(), error


@torch.no_grad()
def capture(model, ids, labels, r, cache_dtype=torch.bfloat16):
    values, hooks = {}, []
    def save(key):
        def hook(module, inputs, output):
            values[key + '_in'], values[key] = inputs[0].detach(), output.detach()
        return hook
    for name, key in ((MODULES[0], 'up'), (MODULES[1], 'down'),
                      ('h.3.rms_2', 'mlp_norm'), ('ln_f', 'final_norm')):
        hooks.append(model.get_submodule(name).register_forward_hook(save(key)))
    try:
        logits = model(ids)
    finally:
        for hook in hooks: hook.remove()
    h, z = values['up_in'].flatten(0, 1), values['up'].flatten(0, 1)
    residual = values['final_norm_in'].flatten(0, 1)
    scale = (residual.square().mean(-1) + model.ln_f.eps).rsqrt()
    q = residual * scale[:, None] / math.sqrt(residual.shape[-1])
    down = model.get_submodule(MODULES[1]).weight.detach()
    gate = gelu_derivative(z)
    # q and scale are fixed normalization-derivative metadata, kept in FP32.
    rq = q @ r.T
    common = dict(structured=True, output_count=1, net_rms=torch.zeros(len(h), device=h.device))
    exact = [dict(common, kind='pile_last_up', h=h.to(cache_dtype),
                  events={'gate': gate.to(cache_dtype), 'q_down': q @ down,
                          'rq': rq, 'scale': scale}, shared={'factor': r @ down}),
             dict(common, kind='pile_last_down', h=values['down_in'].flatten(0, 1).to(cache_dtype),
                  events={'q': q, 'rq': rq, 'scale': scale}, shared={'factor': r})]
    # Exact derivative of each token's CE. The final MLP has no cross-token
    # descendants, so differentiating the sum yields those per-token gradients.
    dl = logits.flatten(0, 1).softmax(-1)
    dl.scatter_add_(1, labels.reshape(-1, 1), -torch.ones_like(labels.reshape(-1, 1), dtype=dl.dtype))
    g = (dl @ model.lm_head.weight) * model.ln_f.weight
    g = scale[:, None] * (g - q * (g * q).sum(-1, keepdim=True))
    gradients = ((g @ down) * gate, g)
    scalar = [dict(common, kind='scalar_loss', h=e['h'],
                   events={'g': grad.to(cache_dtype)}, shared={}) for e, grad in zip(exact, gradients)]
    tail = {'h': h, 'prefix': values['mlp_norm_in'].flatten(0, 1),
            'labels': labels.reshape(-1), 'logits': logits.flatten(0, 1)}
    return {'output': exact, 'scalar': scalar}, tail


def scores(entry, pieces):
    """Analytic Jacobian-vector norm with a dense, full-rank output metric."""
    events, factor = entry['events'], entry['shared']['factor']
    auto = torch.is_autocast_enabled(pieces.read.device.type)
    h = entry['h'] if auto else entry['h'].to(pieces.read.dtype)
    reads = h @ pieces.read
    if entry['kind'] == 'pile_last_up':
        gate = events['gate'].to(pieces.write.dtype)
        # Store this large per-token activation in the autocast dtype under mixed precision.
        dtype = torch.get_autocast_dtype(pieces.write.device.type) if auto else pieces.write.dtype
        local = gate.to(dtype).unsqueeze(-1) * pieces.write.to(dtype).unsqueeze(0)
        base = factor @ local
        radial = (events['q_down'].unsqueeze(1) @ local).squeeze(1)
    else:
        base = factor @ pieces.write
        radial = events['q'] @ pieces.write
    # Subtract vectors before squaring rather than subtracting quadratic forms.
    # These residual/norm reductions remain FP32 under mixed precision.
    effect = base.float() - events['rq'].float().unsqueeze(-1) * radial.float().unsqueeze(-2)
    norm = effect.square().sum(-2).add(1e-30).sqrt() * events['scale'][:, None]
    return reads.float().abs() * norm


def tail_forward(model, tail, pieces, masks):
    z = core.masked_output(tail['h'], pieces[0], masks[0])
    mlp = core.masked_output(F.gelu(z, approximate='tanh'), pieces[1], masks[1])
    return model.lm_head(model.ln_f(tail['prefix'] + mlp))
