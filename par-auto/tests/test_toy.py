import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT.parent):
    sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_target import ResidualToy, ToyConfig, sample  # noqa: E402


def small():
    torch.manual_seed(37)
    cfg = ToyConfig(n_features=5, d_embed=7, d_mlp=3)
    target = ResidualToy(cfg).double().requires_grad_(False)
    return target, cfg


def test_cache_holds_exact_write_side_jacobians():
    target, cfg = small()
    x = sample(16, cfg.n_features, torch.Generator().manual_seed(1), "cpu", probability=0.4).double()
    cache = toy.cache_events(target, x, chunk=5)
    assert torch.allclose(cache["output"], target(x), atol=1e-12)
    names = [name for name, _ in toy.matrices(target)]
    assert cache["matrices"]["layer0.mlp_in"]["side"] == "g"
    assert cache["matrices"]["layer0.mlp_out"]["side"] == "g"
    assert cache["matrices"]["layer0.mlp_in"]["cot"].shape == (16, 5, 3)
    assert cache["matrices"]["layer0.mlp_out"]["cot"].shape == (16, 5, 7)
    # Jacobian of the output with respect to the last write is exactly W_U^T.
    last = cache["matrices"][names[-1]]
    g = last["cot"] @ last["output_basis"].T
    assert torch.allclose(g, target.W_U.T.expand(16, -1, -1), atol=1e-12)
    # Finite differences on the first pre-activation.
    first = cache["matrices"]["layer0.mlp_in"]
    delta = torch.zeros(3, dtype=torch.float64)
    delta[1] = 1e-6

    def perturbed(x_row, bump):
        h = x_row @ target.W_E
        pre = h @ target.layers[0].mlp_in.weight.T + bump
        h = h + pre.relu() @ target.layers[0].mlp_out.weight.T
        h = h + target.layers[1](h)
        return h @ target.W_U

    numerical = (perturbed(x[:1], delta) - perturbed(x[:1], -delta)) / 2e-6
    assert torch.allclose(numerical[0], first["cot"][0, :, 1], atol=1e-6)


def test_first_order_prediction_is_exact_for_the_final_matrix():
    target, cfg = small()
    x = sample(12, cfg.n_features, torch.Generator().manual_seed(2), "cpu", probability=0.4).double()
    cache = toy.cache_events(target, x)
    pieces = [core.initialize_pieces(w, 13) for _, w in toy.matrices(target)]
    keeps = [torch.ones(12, 13, dtype=torch.float64) for _ in pieces]
    keeps[3] = core.topk_mask(torch.rand(12, 13), 1).double()
    masked, _, _ = toy.forward_with_pieces(target, x, pieces, keeps)
    predicted = toy.first_order_prediction(cache, pieces, keeps, target, batch=5)
    assert torch.allclose(masked - cache["output"], predicted, atol=1e-12)


def test_free_pieces_all_kept_and_jacobian_compression():
    torch.manual_seed(27)
    target = ResidualToy(ToyConfig(n_features=5, d_embed=20, d_mlp=3)).double().requires_grad_(False)
    x = torch.randn(7, 5, dtype=torch.float64)
    pieces = [core.initialize_pieces(w, 23) for _, w in toy.matrices(target)]
    keeps = [torch.ones(7, p.count, dtype=torch.float64) for p in pieces]
    actual, _, _ = toy.forward_with_pieces(target, x, pieces, keeps)
    torch.testing.assert_close(actual, target(x), atol=1e-12, rtol=1e-12)
    cache = toy.cache_events(target, x, chunk=3)
    leaf = x.clone().requires_grad_()
    output, _, outputs = toy.forward_with_pieces(target, leaf)
    jac = torch.stack([torch.autograd.grad(output[:, j].sum(), outputs[1], retain_graph=True)[0]
                       for j in range(5)], dim=1)
    entry = cache["matrices"]["layer0.mlp_out"]
    assert entry["output_basis"].shape == (20, 8)
    torch.testing.assert_close(entry["cot"] @ entry["output_basis"].T, jac, atol=1e-12, rtol=1e-12)


