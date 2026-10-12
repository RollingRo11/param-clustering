import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT.parent):
    sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_evaluate as experiment  # noqa: E402
import parauto_method as obj  # noqa: E402


def test_lp_anti_splitting_and_finite_gradients_at_zero():
    a = torch.tensor([[0.1]])
    split = torch.tensor([[0.05, 0.05]])
    assert obj.lp_concentration(split, 0.5, 1e-6) > obj.lp_concentration(a, 0.5, 1e-6)
    torch.testing.assert_close(obj.lp_concentration(split, 1., 1e-6), obj.lp_concentration(a, 1., 1e-6))
    zero = torch.zeros(3, 5, requires_grad=True)
    value = obj.lp_concentration(zero, 0.5, 1e-6).sum()
    value.backward()
    assert value == 0
    assert torch.isfinite(zero.grad).all()


def test_threshold_calibration_does_not_assume_monotonic_accuracy():
    curve = [{"threshold_rms": 0.01, "pass_meaningful": 0.96, "kept_mean_meaningful": 60},
             {"threshold_rms": 0.02, "pass_meaningful": 0.91, "kept_mean_meaningful": 40},
             {"threshold_rms": 0.03, "pass_meaningful": 0.95, "kept_mean_meaningful": 30},
             {"threshold_rms": -1., "pass_meaningful": 1., "kept_mean_meaningful": 100}]
    chosen = experiment.select_operating_points(curve)
    assert chosen["pass_0.95"]["threshold_rms"] == 0.03
    assert chosen["pass_0.99"]["threshold_rms"] == -1.
    assert chosen["budget_32"]["threshold_rms"] == 0.03
    with pytest.raises(ValueError, match="meaningful"):
        experiment.select_operating_points([{"pass_meaningful": None}])


def test_high_cancellation_update_matches_plain_adam_then_projection():
    w = torch.ones(1, 1, dtype=torch.float64)
    initial = core.FreePieces(torch.ones(1, 3, dtype=torch.float64),
                              torch.tensor([[100., -100., 1.]], dtype=torch.float64))
    entry = {"h": torch.ones(3, 1, dtype=torch.float64), "cot": torch.ones(3, 1, 1, dtype=torch.float64), "output_basis": None}
    cfg = obj.TrainConfig(steps=1, batch=3, log_interval=1)
    expected = obj.clone_pieces(initial)
    expected.read.requires_grad_(); expected.write.requires_grad_()
    optimizer = torch.optim.Adam([expected.read, expected.write], lr=cfg.lr, betas=(.9, .99))
    reference = obj.concentration(obj.scores_of(entry, initial), cfg, 1.).mean()
    loss = obj.concentration(obj.scores_of(entry, obj.project(w, expected, cfg)), cfg, 1.).mean()/reference
    loss.backward(); optimizer.step(); obj.project_(w, expected, cfg)
    fitted, history = obj.fit_matrix(entry, w, initial, cfg, log=lambda _: None)
    torch.testing.assert_close(fitted.read, expected.read, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(fitted.write, expected.write, atol=1e-12, rtol=1e-12)
    assert not torch.equal(fitted.read, initial.read)
    assert obj.scores_of(entry, fitted).sum(-1).mean() > 100
    assert core.reconstruction_error(w, fitted) < 1e-10
    assert history[-1]['optimizer_updates'] == 1
