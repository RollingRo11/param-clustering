"""Downstream metric factors for the layer-local term: M = E[g g^T] on a calibration batch, top eigenvectors.

The raw squared local error asks the kept set to reproduce parts of a layer's output that nothing
downstream ever reads. M weights the error by what downstream actually reads ON AVERAGE, so error may
hide in directions no later computation uses, but not in directions this token happens not to use.

g is the cotangent at the site where the local error is measured:
  q/k/v/o/down_proj   that matrix's own output;
  mlp.c_fc            the GELU output, which for this target is exactly cot(down_proj output) @ W_down,
                      because down_proj is linear and reads the GELU output directly.
Factors L satisfy M = L L^T with L = V sqrt(lambda) over the top `rank` eigenvectors, so the training
term is ||L^T e||^2. Sampled next tokens come from the model itself, as in fisher_attribution.
"""
import torch

from common import MODULES
from fisher_attribution import capture_fisher


def metric_factors(model, batches, k=8, rank=64, seed=9_000_001, microbatch=2):
    """batches: an iterable of token row tensors. Returns ({name: [d_out, rank]}, diagnostics)."""
    down = {i: model.get_submodule(f'h.{i}.mlp.down_proj').weight.detach().float() for i in range(4)}
    totals, samples = {n: None for n in MODULES}, 0
    for j, rows in enumerate(batches):
        # capture_fisher runs its own backward passes, so it must not be called under no_grad.
        entries = capture_fisher(model, rows, k=k, seed=seed + j, microbatch=microbatch)
        with torch.no_grad():
            cot = {n: e['events']['cot'].detach() for n, e in zip(MODULES, entries)}
            for name in MODULES:
                layer = int(name.split('.')[1])
                g = (cot[f'h.{layer}.mlp.down_proj'].float() @ down[layer]) if name.endswith('c_fc') else cot[name].float()
                g = g.flatten(0, 1)                              # [tokens*k, d_out]
                product = g.T @ g
                totals[name] = product if totals[name] is None else totals[name] + product
                del g, product
            samples += cot[MODULES[0]].shape[0] * k
        del entries, cot
    factors, diagnostics = {}, {}
    with torch.no_grad():
        for name, total in totals.items():
            moment = (total / samples).double()
            values, vectors = torch.linalg.eigh(moment)
            top = values.flip(0)[:rank].clamp_min(0)
            factors[name] = (vectors.flip(1)[:, :rank] * top.sqrt()).float()
            diagnostics[name] = dict(dimension=int(moment.shape[0]), rank=int(top.numel()),
                                     trace_share=float(top.sum()/values.clamp_min(0).sum().clamp_min(1e-30)),
                                     site='gelu_output' if name.endswith('c_fc') else 'matrix_output')
    return factors, dict(samples=samples, fisher_k=k, rank=rank, per_matrix=diagnostics)
