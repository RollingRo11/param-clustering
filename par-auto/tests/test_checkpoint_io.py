import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT.parent):
    sys.path.insert(0, str(directory))

import parauto_core as core  # noqa: E402
import parauto_io as checkpoints  # noqa: E402
import parauto_toy as toy  # noqa: E402
from parauto_target import ResidualToy, ToyConfig  # noqa: E402


def test_current_and_historical_checkpoints_load_the_same_factors(tmp_path):
    torch.manual_seed(19)
    target = ResidualToy(ToyConfig(n_features=5, d_embed=9, d_mlp=3)).double()
    pieces = [core.initialize_pieces(w, 13) for _, w in toy.matrices(target)]
    records = [{"read": p.read, "write": p.write} for p in pieces]
    torch.save({"pieces": records}, tmp_path / "pieces.pt")
    current = checkpoints.load_pieces(tmp_path, target, "cpu")
    (tmp_path / "pieces.pt").unlink()
    legacy = tmp_path / "B_lp"
    legacy.mkdir()
    for (name, _), record in zip(toy.matrices(target), records, strict=True):
        torch.save(record, legacy / f"pieces_{name}.pt")
    historical = checkpoints.load_pieces(tmp_path, target, "cpu")
    for a, b in zip(current, historical, strict=True):
        assert torch.equal(a.read, b.read) and torch.equal(a.write, b.write)
    records[0]["write"] = torch.zeros_like(records[0]["write"])
    torch.save({"pieces": records}, tmp_path / "pieces.pt")
    with pytest.raises(ValueError, match="matrix sum mismatch"):
        checkpoints.load_pieces(tmp_path, target, "cpu")
