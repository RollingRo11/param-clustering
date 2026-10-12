import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT.parent):
    sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_toy as toy  # noqa: E402
import plot_spd_comparison as comparison  # noqa: E402
from parauto_target import ResidualToy, ToyConfig  # noqa: E402


def test_contributions_match_explicit_equations_and_sum():
    torch.manual_seed(24)
    target = ResidualToy(ToyConfig(n_features=4, d_embed=7, d_mlp=3)).double()
    pieces = [core.initialize_pieces(w, 11) for _, w in toy.matrices(target)]
    reference, candidates = comparison.contribution_profiles(target, pieces)
    for layer, module in enumerate(target.layers):
        inp, out = pieces[2*layer:2*layer+2]
        for feature in range(4):
            e, u = target.W_E[feature], target.W_U[:, feature]
            torch.testing.assert_close(reference[layer, feature], (u @ module.mlp_out.weight) * (module.mlp_in.weight @ e))
            for c in range(11):
                piece = torch.outer(inp.write[:, c], inp.read[:, c])
                expected = (u @ (out.write @ out.read.T)) * (piece @ e)
                torch.testing.assert_close(candidates[layer, feature, c], expected)
    torch.testing.assert_close(candidates.sum(dim=2), reference, atol=1e-12, rtol=1e-12)


def test_selection_is_one_piece_per_feature_not_pointwise_or_absolute_max():
    candidates = torch.tensor([[[[2., -9., 0.], [1., 1., 3.], [-20., 0., 0.]],
                                [[4., -5., 0.], [1., 3., 0.], [0., 0., 0.]]]])
    matched, index = comparison.select_like_paper(candidates)
    assert index.tolist() == [[1, 0]]
    torch.testing.assert_close(matched, torch.tensor([[[1., 1., 3.], [4., -5., 0.]]]))
    assert not torch.equal(matched, candidates.max(dim=2).values)
