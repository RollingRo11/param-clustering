import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT.parent):
    sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_method as obj  # noqa: E402
import parauto_structured as structured  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_target import ResidualToy, ToyConfig  # noqa: E402


@pytest.mark.parametrize("features,width", [(7, 3), (2, 4)])
def test_structured_scores_gradients_and_audit_match_full_jacobians(features, width):
    torch.manual_seed(51)
    target = ResidualToy(ToyConfig(n_features=features, d_embed=19, d_mlp=width)).double().requires_grad_(False)
    x = torch.randn(13, features, dtype=torch.float64)
    x[0] = 0  # Includes the all-inactive ReLU boundary.
    dense = toy.cache_events(target, x, chunk=5)
    compact = structured.cache_events(target, x, chunk=5)
    torch.testing.assert_close(compact["output"], dense["output"], atol=1e-12, rtol=1e-12)
    for name, weight in toy.matrices(target):
        a, b = [cache["matrices"][name] for cache in (dense, compact)]
        pieces = core.initialize_pieces(weight, 23)
        pieces.read.requires_grad_()
        pieces.write.requires_grad_()
        pa = core.projected_pieces(weight, pieces)
        sa, sb = [obj.scores_of(entry, pa) for entry in (a, b)]
        torch.testing.assert_close(sa, sb, atol=1e-12, rtol=1e-10)
        ga = torch.autograd.grad(obj.lp_concentration(sa, 0.5, 1e-6).mean(), (pieces.read, pieces.write), retain_graph=True)
        gb = torch.autograd.grad(obj.lp_concentration(sb, 0.5, 1e-6).mean(), (pieces.read, pieces.write))
        for left, right in zip(ga, gb, strict=True):
            torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(obj.net_effect_rms(a, weight), obj.net_effect_rms(b, weight), atol=1e-12, rtol=1e-10)
        indices = torch.tensor([0, 7, 2, 7])
        torch.testing.assert_close(obj.scores_of(obj.slice_entry(b, indices), pieces), obj.scores_of(b, pieces)[indices])
        cfg = obj.TrainConfig(steps=4, batch=8, normalization_events=13, log_interval=4)
        fitted, _ = obj.fit_matrix(b, weight, pieces, cfg, log=lambda row: None)
        assert core.reconstruction_error(weight, fitted) < 1e-10


def test_null_writes_remain_finite_and_no_spurious_effect():
    torch.manual_seed(72)
    target = ResidualToy(ToyConfig(n_features=5, d_embed=19, d_mlp=3)).double().requires_grad_(False)
    x = torch.randn(11, 5, dtype=torch.float64)
    compact = structured.cache_events(target, x)
    f = target.W_U.T
    null = torch.linalg.svd(f, full_matrices=True).Vh[5:].T
    entry = compact["matrices"]["layer1.mlp_out"]
    pieces = core.FreePieces(torch.randn(3, null.shape[1], dtype=x.dtype), null.clone().requires_grad_())
    scores = obj.scores_of(entry, pieces)
    assert scores.abs().max() < 1e-14
    gradient, = torch.autograd.grad(obj.lp_concentration(scores, .5, 1e-6).mean(), (pieces.write,))
    assert torch.isfinite(gradient).all()
