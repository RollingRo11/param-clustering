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
from typing import Any, Callable

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
        # Custom autograd Functions are autocast boundaries. Residual paths can
        # therefore hand backward a mixture of fp32 and bf16 tensors even when
        # q/k/v came from autocast linears. Use one compute dtype, then return
        # each input gradient in that input's original dtype.
        low_precision = {torch.float16, torch.bfloat16}
        compute_dtype = (q.dtype if q.dtype in low_precision else torch.float32)
        qb = q.to(compute_dtype)
        kb = k.to(compute_dtype)
        vb = v.to(compute_dtype)
        go = grad_out.to(compute_dtype)
        scores = (torch.matmul(qb, kb.transpose(-2, -1))
                  * ctx.scaling).float()
        if ctx.has_mask:
            scores = scores + stored_mask.float()
        elif ctx.is_causal:
            causal = torch.ones(scores.shape[-2:], dtype=torch.bool,
                                device=scores.device).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))

        p = torch.softmax(scores, dim=-1).to(compute_dtype)
        pt = torch.softmax(scores / ctx.tau, dim=-1).to(compute_dtype)
        dp = torch.matmul(go, vb.transpose(-2, -1))
        ds = (dp - (dp * pt).sum(-1, keepdim=True)) * pt

        # Grad norm from GIM: the Q/K product jointly receives half the credit
        # (split equally), and the attention/value product receives the other
        # half.  repeat_kv outside this Function sums grouped-head gradients.
        dq = (torch.matmul(ds, kb) * (ctx.scaling / 4.0)).to(q.dtype)
        dk = (torch.matmul(ds.transpose(-2, -1), qb)
              * (ctx.scaling / 4.0)).to(k.dtype)
        dv = (torch.matmul(p.transpose(-2, -1), go) / 2.0).to(v.dtype)
        return dq, dk, dv, None, None, None, None


class _HalfBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad / 2.0


def install_llama_gim(target: torch.nn.Module, tau: float = 2.0,
                      *, unpadded_causal: bool = False) -> None:
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
        if module._gim_unpadded_causal:
            # The collector passes only dense, unpadded token rows. Recent HF
            # versions nevertheless materialize a 4-D causal mask, while
            # PyTorch Flash SDPA requires the equivalent null-mask/is_causal
            # representation.
            attention_mask = None
            causal = query.shape[2] > 1
        else:
            causal = (query.shape[2] > 1 and attention_mask is None
                      and (module.is_causal if is_causal is None else is_causal))
        out = _GimSDPA.apply(query, key, value, attention_mask,
                             module.scaling if scaling is None else scaling,
                             causal, module._gim_tau)
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
        elif name == "LlamaAttention":
            module._gim_tau = float(tau)
            module._gim_unpadded_causal = bool(unpadded_causal)

    # The wrapper stores the actual HF module as ``hf``.
    hf = getattr(target, "hf", target)
    hf.config._attn_implementation = "gim_flash_sdpa"


def linear_kernel(enabled: bool,
                  mode: str = "default") -> Callable:
    """Return a pure eager or compiled linear kernel for :class:`Capture`.

    The cache-writing wrapper must remain outside this boundary. A whole-model
    compiled graph cannot expose gradients with respect to sibling captured
    outputs, while compiling a side-effecting capture closure loses its cache
    tensors. A pure ``F.linear`` boundary preserves both autograd connectivity
    and Inductor cache reuse across the repeated Llama projection shapes.
    ``reduce-overhead`` is intentionally not the default: its CUDA graphs wait
    for backward calls that do not exist for every captured output and trigger
    repeated graph capture in this workload.
    """
    if not enabled:
        return F.linear
    torch.set_float32_matmul_precision("high")
    return torch.compile(
        F.linear, mode=mode, fullgraph=True, dynamic=False)
