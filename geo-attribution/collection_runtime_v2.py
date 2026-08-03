"""Mixed-dtype-safe overlay for the GIM runtime."""

from __future__ import annotations

import torch

import collection_runtime as _base
from collection_runtime import (compile_model, file_sha256, install_llama_gim,
                                model_pass, stable_fingerprint)


def _backward(ctx, grad_out):
    q, k, v, stored_mask = ctx.saved_tensors
    # Custom autograd Functions are autocast boundaries, so residual paths can
    # hand us a mixture of fp32 and bf16 tensors.  Use one tensor-core compute
    # dtype, then return each input gradient in that input's original dtype.
    compute_dtype = torch.bfloat16 if q.is_cuda else q.dtype
    qb, kb, vb = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
    go = grad_out.to(compute_dtype)
    scores = (torch.matmul(qb, kb.transpose(-2, -1)) * ctx.scaling).float()
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
    dq = (torch.matmul(ds, kb) * (ctx.scaling / 4.0)).to(q.dtype)
    dk = (torch.matmul(ds.transpose(-2, -1), qb)
          * (ctx.scaling / 4.0)).to(k.dtype)
    dv = (torch.matmul(p.transpose(-2, -1), go) / 2.0).to(v.dtype)
    return dq, dk, dv, None, None, None, None


_base._GimSDPA.backward = staticmethod(_backward)

__all__ = ["compile_model", "file_sha256", "install_llama_gim", "model_pass",
           "stable_fingerprint"]
