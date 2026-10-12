import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT.parent):
    sys.path.insert(0, str(directory))

import inspect_components as inspect  # noqa: E402
import parauto_core as core  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_target import ResidualToy, ToyConfig  # noqa: E402


def test_batched_single_piece_ablation_matches_explicit_kept_piece_sum():
    torch.manual_seed(46)
    target = ResidualToy(ToyConfig(n_features=4, d_embed=7, d_mlp=3)).double().requires_grad_(False)
    x = torch.randn(9, 4, dtype=torch.float64)
    pieces = [core.initialize_pieces(w, 11) for _, w in toy.matrices(target)]
    for which, p in enumerate(pieces):
        indices = torch.tensor([0, 3, 8])
        batched = inspect.ablated_outputs(target, x, p, which, indices)
        for i, c in enumerate(indices):
            keeps = [torch.ones(len(x), q.count, dtype=torch.float64) for q in pieces]
            keeps[which][:, c] = 0
            expected = toy.forward_with_pieces(target, x, pieces, keeps)[0]
            torch.testing.assert_close(batched[i], expected, atol=1e-12, rtol=1e-12)


def test_feature_specific_and_shared_profile_counts():
    profile = torch.cat([torch.eye(5), 0.5 * torch.ones(5, 1)], dim=1)
    result = inspect.describe_profile(profile, torch.ones(5))
    assert result["features_with_strong_selective_piece"] == 5
    assert result["strong_selective_pieces"] == 5
    assert result["median_top_piece_purity"] == 1
    shared = inspect.describe_profile(torch.ones(5, 3), torch.ones(5))
    assert shared["features_with_strong_selective_piece"] == 0
