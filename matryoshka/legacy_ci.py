"""Restore the Feb-2026 (commit 28f6436) GlobalSharedTransformerCiFn input path: no RMSNorm on inputs."""
import torch
from param_decomp.models import components as C

def _old_forward(self, input_acts):
    concatenated = torch.cat([input_acts[name] for name in self.layer_order], dim=-1)
    x = self._input_projector(concatenated)
    assert x.ndim == 3
    for block in self._blocks:
        x = block(x)
    output = self._output_head(x)
    split_outputs = torch.split(output, self.split_sizes, dim=-1)
    return {name: split_outputs[i] for i, name in enumerate(self.layer_order)}

C.GlobalSharedTransformerCiFn.forward = _old_forward
