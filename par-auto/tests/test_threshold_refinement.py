import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parauto_refine as refine  # noqa: E402


def test_budget_threshold_uses_only_calibration_meaningful_rows():
    scores = [torch.tensor([[1., 2., 3., 4., 5.], [1000., 2000., 3000., 4000., 5000.], [6., 7., 8., 9., 10.]]),
              torch.tensor([[11., 12., 13., 14., 15.], [1000., 2000., 3000., 4000., 5000.], [16., 17., 18., 19., 20.]])]
    meaningful = torch.tensor([True, False, True])
    thresholds = refine.budget_thresholds(scores, meaningful, (0, 3, 10))
    for budget, tau in thresholds.items():
        count = sum((s[meaningful] > tau).sum(-1).float() for s in scores).mean()
        assert count == budget
    with pytest.raises(ValueError, match="meaningful"):
        refine.budget_thresholds(scores, torch.zeros(3, dtype=torch.bool))


def test_tied_scores_report_attainable_budget_without_inventing_tie_breaking():
    scores = [torch.ones(2, 5)]
    tau = refine.budget_thresholds(scores, torch.ones(2, dtype=torch.bool), (2,))[2]
    assert tau == 1.
    assert (scores[0] > tau).sum() == 0
