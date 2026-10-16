"""GIM sensor for the upstream 67M target's eager attention and RMSNorm.

Rules follow corticph/gim commit 7e31c65f437aaa2bb33dcd920f09b0d35fd6802a:
T=2 softmax VJP (no extra 1/T), Q/K gradients /4, V /2, frozen RMS.
This target has a plain GELU MLP: the gated-MLP rule does not apply.
Forward values and weights are unchanged. The returned g is a surrogate,
not the true CE derivative. Patches exist only during cache collection.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import torch.nn.functional as F

RULES = dict(softmax_temperature=2., q_scale=.25, k_scale=.25, v_scale=.5,
             freeze_rms=True, mlp='ordinary GELU backward; no multiplicative gate',
             reference='https://github.com/corticph/gim/tree/7e31c65f437aaa2bb33dcd920f09b0d35fd6802a')
_SOFTMAX = F.softmax


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class TemperatureSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, temperature):
        ctx.save_for_backward(x)
        ctx.dim, ctx.temperature = dim, temperature
        return _SOFTMAX(x, dim=dim)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        # FP32 reductions for BF16; preserve float64 in analytic tests.
        work = x.double() if x.dtype == torch.float64 else x.float()
        prob = _SOFTMAX(work / ctx.temperature, dim=ctx.dim)
        g = grad.to(work.dtype)
        result = prob * (g - (g * prob).sum(ctx.dim, keepdim=True))
        return result.to(x.dtype), None, None


def frozen_rms_forward(self, hidden_states):
    dtype = hidden_states.dtype
    x = hidden_states.float()
    inverse = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    return self.weight * (x * inverse).to(dtype)


@contextmanager
def gim_context(model, *, temperature=2., freeze_rms=True):
    """Single-threaded capture scope; restores all hooks/functions on errors."""
    from param_decomp.pretrain.models.llama_simple_mlp import CausalSelfAttention, LlamaRMSNorm
    attentions = [m for m in model.modules() if isinstance(m, CausalSelfAttention)]
    norms = [m for m in model.modules() if isinstance(m, LlamaRMSNorm)]
    if not attentions or not norms:
        raise ValueError('GIM adapter expects upstream LlamaSimpleMLP')
    if any(m.flash_attention or not m.use_grouped_query_attention for m in attentions):
        raise ValueError('GIM adapter requires eager attention with separate Q/K/V projections')
    original_softmax = F.softmax
    if original_softmax is not _SOFTMAX:
        raise RuntimeError('Do not nest GIM or other softmax patches')
    forwards = []
    calls = dict(softmax=0, norms=len(norms) if freeze_rms else 0)
    def softmax(input, dim=None, _stacklevel=3, dtype=None):
        calls['softmax'] += 1
        if dim is None: raise ValueError('GIM requires an explicit softmax dimension')
        # Neutral rule must use the original saved BF16 probabilities/VJP.
        # Recomputing even a T=1 softmax in FP32 changes rounding near zero.
        if temperature == 1.:
            return original_softmax(input, dim=dim, dtype=dtype)
        return TemperatureSoftmax.apply(input if dtype is None else input.to(dtype), dim, temperature)
    try:
        F.softmax = softmax
        if freeze_rms:
            for norm in norms:
                forwards.append((norm, norm.__dict__.get('forward')))
                norm.forward = MethodType(frozen_rms_forward, norm)
        yield calls
    finally:
        F.softmax = original_softmax
        for norm, forward in forwards:
            if forward is None: del norm.forward
            else: norm.forward = forward


def capture_gim(model, rows, microbatch=4, *, temperature=2., q_scale=.25,
                k_scale=.25, v_scale=.5, freeze_rms=True):
    from train_pair import capture
    counts = dict(q_proj=0, k_proj=0, v_proj=0)
    scales = dict(q_proj=q_scale, k_proj=k_scale, v_proj=v_scale)
    def transform(name, output):
        suffix = name.split('.')[-1]
        if suffix not in scales: return output
        counts[suffix] += 1
        return ScaleGradient.apply(output, scales[suffix])
    with gim_context(model, temperature=temperature, freeze_rms=freeze_rms) as calls:
        entries = capture(model, rows, microbatch=microbatch, output_transform=transform)
    expected = len(model.h) * ((len(rows)+microbatch-1)//microbatch)
    if calls['softmax'] != expected or any(v != expected for v in counts.values()):
        raise RuntimeError(f'GIM rules did not cover all attention calls: {calls}, {counts}')
    for entry in entries:
        entry['shared']['sensor'] = 'gim'
    return entries
