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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_one_loss_matches_separate_fits_and_uses_one_update(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA validation requires a GPU allocation")
    torch.manual_seed(85)
    target = ResidualToy(ToyConfig(n_features=7, d_embed=19, d_mlp=3)).to(device).double().requires_grad_(False)
    x = torch.randn(29, 7, device=device, dtype=torch.float64)
    cache = structured.cache_events(target, x)
    pairs = toy.matrices(target)
    weights = [weight for _, weight in pairs]
    entries = [cache["matrices"][name] for name, _ in pairs]
    initial = [core.initialize_pieces(weight, 23) for weight in weights]
    cfg = obj.TrainConfig(steps=8, batch=11, normalization_events=17,
                              log_interval=1, tau=.01)
    counts = {"backward": 0, "step": 0}
    original_backward, original_step = torch.Tensor.backward, torch.optim.Adam.step

    def backward(self, *args, **kwargs):
        counts["backward"] += 1
        return original_backward(self, *args, **kwargs)

    def step(self, *args, **kwargs):
        counts["step"] += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", backward)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    separate = [obj.fit_matrix(entry, weight, item, cfg, log=lambda row: None)
                for entry, weight, item in zip(entries, weights, initial, strict=True)]
    assert counts == {"backward": cfg.steps * 4, "step": cfg.steps * 4}
    counts.update(backward=0, step=0)
    joint, histories = obj.fit(entries, weights, initial, cfg, log=lambda row: None)
    assert counts == {"backward": cfg.steps, "step": cfg.steps}
    for i, ((expected, old_history), actual, rows) in enumerate(zip(separate, joint, histories, strict=True)):
        torch.testing.assert_close(actual.read, expected.read, atol=1e-11, rtol=1e-9)
        torch.testing.assert_close(actual.write, expected.write, atol=1e-11, rtol=1e-9)
        assert core.reconstruction_error(weights[i], actual) < 1e-10
        for old, row in zip(old_history, rows, strict=True):
            assert old.keys() == row.keys()
            for key in row:
                assert row[key] == pytest.approx(old[key], rel=1e-8, abs=1e-11)
