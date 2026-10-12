import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parauto_core as core
import parauto_method as method
import parauto_scalar as scalar
import parauto_structured as structured
import parauto_toy as toy
from parauto_target import ResidualToy, ToyConfig


def test_scalar_cache_matches_gate_derivatives_partition_and_chunking():
    torch.manual_seed(921)
    target = ResidualToy(ToyConfig(n_features=7, d_embed=13, d_mlp=3)).double().requires_grad_(False)
    x = torch.randn(9, 7, dtype=torch.float64)
    x[0] = 0
    cache = scalar.cache_events(target, x, chunk=4)
    other = scalar.cache_events(target, x, chunk=9)
    pieces = [core.initialize_pieces(w, 17) for _, w in toy.matrices(target)]
    gates = [torch.ones(len(x), p.count, dtype=x.dtype, requires_grad=True) for p in pieces]
    output = toy.forward_with_pieces(target, x, pieces, gates)[0]
    gate_grads = torch.autograd.grad(scalar.task_loss(output, x).sum(), gates)
    for (name, weight), part, expected in zip(toy.matrices(target), pieces, gate_grads, strict=True):
        entry = cache['matrices'][name]
        signed = scalar.signed_scores(entry, part)
        torch.testing.assert_close(signed, expected, atol=1e-11, rtol=1e-9)
        torch.testing.assert_close(method.scores_of(entry, part), expected.abs(), atol=1e-11, rtol=1e-9)
        net = ((entry['h'] @ weight.T) * entry['events']['g']).sum(-1)
        torch.testing.assert_close(signed.sum(-1), net, atol=1e-11, rtol=1e-9)
        torch.testing.assert_close(entry['net_rms'], net.abs())
        torch.testing.assert_close(entry['events']['g'], other['matrices'][name]['events']['g'])
        assert signed[0].count_nonzero() == 0
        index = torch.tensor([3, 0, 3, 7])
        torch.testing.assert_close(method.scores_of(method.slice_entry(entry, index), part), signed[index].abs())


def test_scalar_projected_fit_preserves_sum_and_applies_every_update():
    torch.manual_seed(922)
    target = ResidualToy(ToyConfig(n_features=5, d_embed=9, d_mlp=3)).double().requires_grad_(False)
    cache = scalar.cache_events(target, torch.randn(23, 5, dtype=torch.float64))
    cfg = method.TrainConfig(steps=3, batch=8, normalization_events=23, log_interval=1)
    named = toy.matrices(target)
    entries = [cache['matrices'][n] for n, _ in named]
    weights = [w for _, w in named]
    initial = [method.prepare_initial(w, 13, cfg, log=lambda _: None)[0] for e, w in zip(entries, weights)]
    fitted, histories = method.fit(entries, weights, initial, cfg, log=lambda _: None)
    for w, p, hist in zip(weights, fitted, histories):
        assert core.reconstruction_error(w, p) < 1e-10
        assert hist[-1]['optimizer_updates'] == cfg.steps

