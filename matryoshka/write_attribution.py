"""Write-size sensor: b_c = |r_c.h| ||u_c||, the norm of one piece's write.

Only the input h of each matrix is needed, so capture is a forward pass with
no loss and no backward. For any removed set S at one matrix,
||sum_{c in S} u_c (r_c.h)|| <= sum_{c in S} b_c on that cached input.
The same microbatching as capture() keeps h bitwise identical to the
gradient and GIM caches, so only the score differs between runs.
"""
import torch
from common import CONTEXT, MODULES


@torch.no_grad()
def capture_inputs(model, rows, microbatch=4):
    entries = None
    for first in range(0, len(rows), microbatch):
        block = rows[first:first+microbatch]
        pre, hooks = {}, []
        def retain(name):
            def hook(_m, inputs, _out): pre[name] = inputs[0]
            return hook
        for name in MODULES: hooks.append(model.get_submodule(name).register_forward_hook(retain(name)))
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(block[:, :-1])
        finally:
            for hook in hooks: hook.remove()
        if entries is None:
            entries = [dict(structured=True, kind='write_norm', shared={'sensor': 'write_norm'}, events={},
                            h=torch.empty(len(rows)*CONTEXT, pre[n].shape[-1], dtype=torch.bfloat16, device='cuda'),
                            net_rms=torch.zeros(len(rows)*CONTEXT, device='cuda')) for n in MODULES]
        lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
        for n, e in zip(MODULES, entries):
            e['h'][lo:hi] = pre[n].flatten(0, 1).bfloat16()
    return entries
