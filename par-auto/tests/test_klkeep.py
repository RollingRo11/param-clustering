import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import parauto_core as core
import parauto_toy as toy
from parauto_target import ResidualToy, ToyConfig
from evaluate_klkeep import removal_ranks, keep_masks, distortions, evaluate_curve


def test_global_order_is_nested_and_removes_exactly_one_component():
    scores = [torch.tensor([[3., 0.], [1., 1.]]), torch.tensor([[1., 2.], [1., 1.]])]
    ranks = removal_ranks(scores)
    assert ranks.tolist() == [[3, 0, 1, 2], [0, 1, 2, 3]]
    masks = torch.cat(keep_masks(ranks, torch.arange(5), [2, 2]), -1)
    assert masks.sum(-1).tolist() == [[4, 4], [3, 3], [2, 2], [1, 1], [0, 0]]
    assert ((masks[:-1].int()-masks[1:].int()).sum(-1) == 1).all()


def test_kl_direction_and_regression_mse():
    ref = torch.tensor([[3., 0., -1.]])
    candidate = torch.tensor([[0., 2., 1.]])
    kl, mse = distortions(ref, candidate)
    p, q = ref.double().softmax(-1), candidate.double().softmax(-1)
    torch.testing.assert_close(kl, (p * (p/q).log()).sum(-1))
    assert not torch.allclose(kl, (q * (q/p).log()).sum(-1))
    torch.testing.assert_close(mse, (candidate.double()-ref).square().mean(-1))
    assert float(distortions(ref, ref)[0]) == 0


def test_batched_curve_matches_independent_explicit_weight_ablation():
    torch.manual_seed(76)
    target = ResidualToy(ToyConfig(n_features=3, d_embed=5, d_mlp=2)).double().requires_grad_(False)
    x = torch.randn(5, 3, dtype=torch.float64)
    parts = [core.initialize_pieces(w, 3) for _, w in toy.matrices(target)]
    ranks = removal_ranks([torch.rand(5, 3) for _ in parts])
    values, _ = evaluate_curve(target, x, parts, ranks, batch=2, level_batch=5, temperature=1.)
    for removed in range(13):
        ys = []
        for event in range(len(x)):
            h = x[event] @ target.W_E
            weights = [(p.write * (ranks[event, i*3:(i+1)*3] >= removed)) @ p.read.T
                       for i, p in enumerate(parts)]
            for i in range(2):
                h = h + (h @ weights[2*i].T).relu() @ weights[2*i+1].T
            ys.append(h @ target.W_U)
        kl, mse = distortions(target(x), torch.stack(ys))
        torch.testing.assert_close(values['kl_target'][:, removed], kl, atol=1e-12, rtol=1e-9)
        torch.testing.assert_close(values['mse_target'][:, removed], mse, atol=1e-12, rtol=1e-9)
