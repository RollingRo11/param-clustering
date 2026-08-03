"""Fast, attribution-aware model-pass utilities for collection.

The model parameters remain fp32.  ``model_pass`` autocasts activations and
linear/attention kernels to bf16, while GIM changes only the backward rules:

* frozen RMSNorm statistics;
* temperature-adjusted softmax gradients (TSG);
* 1/4, 1/4, 1/2 credit through Q, K, V respectively; and
* equal credit across the two multiplicative branches of a gated MLP.

The GIM attention forward is still PyTorch Flash SDPA.  Its custom backward
recomputes the attention probabilities because TSG is intentionally different
from the derivative of the forward softmax.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def stable_fingerprint(payload: dict[str, Any]) -> str:
    """A short, deterministic id for reusable collection artifacts."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def file_sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


@contextmanager
def model_pass(device: str | torch.device, bf16: bool, fused_attention: bool):
    """Autocast one complete forward/backward graph construction.

    Autocast is deliberately used rather than converting the module: parameter
    storage therefore stays fp32, but CUDA kernels and saved activations use
    bf16.  SDPA is restricted to Flash Attention in the optimized setup so a
    silent eager/math fallback cannot invalidate a timing run.
    """
    device = torch.device(device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
    if fused_attention:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        attention = nullcontext()
    with attention, amp:
        yield


class _GimSDPA(torch.autograd.Function):
    """Flash-SDPA forward with the GIM attention backward."""

    @staticmethod
    def forward(ctx, q, k, v, attention_mask, scaling, is_causal, tau):
        ctx.scaling = float(scaling)
        ctx.is_causal = bool(is_causal)
        ctx.tau = float(tau)
        ctx.has_mask = attention_mask is not None
        mask = attention_mask if attention_mask is not None else q.new_empty(0)
        ctx.save_for_backward(q, k, v, mask)
        # The surrounding model_pass pins this call to the Flash backend.
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0,
            scale=ctx.scaling, is_causal=ctx.is_causal)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, stored_mask = ctx.saved_tensors
        # Matmuls remain bf16 tensor-core operations; only the normalization is
        # promoted to fp32.  This mirrors HF's eager attention numerics.
        scores = torch.matmul(q, k.transpose(-2, -1)) * ctx.scaling
        scores = scores.float()
        if ctx.has_mask:
            scores = scores + stored_mask.float()
        elif ctx.is_causal:
            causal = torch.ones(scores.shape[-2:], dtype=torch.bool,
                                device=scores.device).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))

        p = torch.softmax(scores, dim=-1).to(q.dtype)
        pt = torch.softmax(scores / ctx.tau, dim=-1).to(q.dtype)
        dp = torch.matmul(grad_out, v.transpose(-2, -1))
        ds = (dp - (dp * pt).sum(-1, keepdim=True)) * pt

        # Grad norm from GIM: the Q/K product jointly receives half the credit
        # (split equally), and the attention/value product receives the other
        # half.  repeat_kv outside this Function sums grouped-head gradients.
        dq = torch.matmul(ds, k) * (ctx.scaling / 4.0)
        dk = torch.matmul(ds.transpose(-2, -1), q) * (ctx.scaling / 4.0)
        dv = torch.matmul(p.transpose(-2, -1), grad_out) / 2.0
        return dq, dk, dv, None, None, None, None


class _HalfBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad / 2.0


def install_llama_gim(target: torch.nn.Module, tau: float = 2.0) -> None:
    """Install the full GIM recipe on a Hugging Face Llama model.

    Forward values are unchanged.  The patch is idempotent and local to the
    supplied model except for registering a named HF attention interface.
    """
    if getattr(target, "_full_gim_applied", False):
        return
    target._full_gim_applied = True

    from transformers.models.llama.modeling_llama import repeat_kv
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def gim_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                 scaling=None, is_causal=None, **kwargs):
        del dropout, kwargs
        if module.num_key_value_groups > 1:
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        causal = (query.shape[2] > 1 and attention_mask is None
                  and (module.is_causal if is_causal is None else is_causal))
        out = _GimSDPA.apply(query, key, value, attention_mask,
                             module.scaling if scaling is None else scaling,
                             causal, tau)
        return out.transpose(1, 2).contiguous(), None

    # Registering is safe when several targets are constructed in one process.
    try:
        ALL_ATTENTION_FUNCTIONS.register("gim_flash_sdpa", gim_sdpa)
    except ValueError:
        pass

    for module in target.modules():
        name = type(module).__name__
        if name == "LlamaRMSNorm":
            def rms_forward(hidden_states, _module=module):
                dtype = hidden_states.dtype
                h = hidden_states.float()
                variance = h.square().mean(-1, keepdim=True)
                scale = torch.rsqrt(variance + _module.variance_epsilon).detach()
                return _module.weight * (h * scale).to(dtype)
            module.forward = rms_forward
        elif name == "LlamaMLP":
            def mlp_forward(x, _module=module):
                gate = _HalfBackward.apply(_module.gate_proj(x))
                up = _HalfBackward.apply(_module.up_proj(x))
                return _module.down_proj(_module.act_fn(gate) * up)
            module.forward = mlp_forward

    # The wrapper stores the actual HF module as ``hf``.
    hf = getattr(target, "hf", target)
    hf.config._attn_implementation = "gim_flash_sdpa"


def compile_model(model: torch.nn.Module, enabled: bool,
                  mode: str = "reduce-overhead") -> torch.nn.Module:
    if not enabled:
        return model
    torch.set_float32_matmul_precision("high")
    return torch.compile(model, mode=mode, fullgraph=False, dynamic=False)

