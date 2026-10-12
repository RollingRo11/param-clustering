"""Constraint, free-writer semantics, and optimization regressions."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parauto_core as core  # noqa: E402


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (5, 5)])
def test_projection_conserves_weight_after_every_update(shape):
    torch.manual_seed(64)
    w = torch.randn(*shape, dtype=torch.float64)
    p = core.initialize_pieces(w, 13)
    for _ in range(20):
        p.read.add_(0.03 * torch.randn_like(p.read))
        p.write.add_(0.03 * torch.randn_like(p.write))
        held = p.write.clone() if shape[0] <= shape[1] else p.read.clone()
        core.project_piece_sum_(w, p)
        assert core.reconstruction_error(w, p) < 1e-12
        assert torch.equal(held, p.write if shape[0] <= shape[1] else p.read)
    # No hidden residual piece; actual rank-one terms sum to the full matrix.
    explicit = torch.stack([torch.outer(p.write[:, c], p.read[:, c]) for c in range(p.count)])
    torch.testing.assert_close(explicit.sum(0), w, atol=1e-12, rtol=1e-12)
    assert all(torch.linalg.matrix_rank(piece) == 1 for piece in explicit)


def test_free_writes_codes_masks_and_compressed_cotangents():
    torch.manual_seed(5)
    w = torch.randn(7, 3, dtype=torch.float64)
    p = core.initialize_pieces(w, 12)
    assert not torch.allclose(p.write, w @ p.read)
    h = torch.randn(11, 3, dtype=torch.float64)
    q = torch.linalg.qr(torch.randn(7, 4, dtype=torch.float64)).Q
    g = torch.randn(11, 4, dtype=torch.float64)
    a = core.free_codes(h, g, p, q)
    explicit = torch.stack([(h @ p.read[:, c]) * (g @ q.T @ p.write[:, c])
                            for c in range(p.count)], dim=1)
    torch.testing.assert_close(a, explicit)
    keep = torch.rand_like(a)  # fractional masks as well as binary
    masked = core.masked_output(h, p, keep)
    change = ((masked - h @ w.T) * (g @ q.T)).sum(1)
    torch.testing.assert_close(change, -(a * (1 - keep)).sum(1))
    torch.testing.assert_close(core.masked_output(h, p, torch.ones_like(a)), h @ w.T)
    jac = torch.randn(11, 5, 4, dtype=torch.float64)
    expected = (h @ p.read).abs() * (jac @ q.T @ p.write).norm(dim=1)
    torch.testing.assert_close(core.free_codes(h, jac, p, q), expected)
    metric = jac.transpose(-1, -2) @ jac
    writes = q.T @ p.write
    via_metric = (h @ p.read).abs() * (writes * (metric @ writes)).sum(1).sqrt()
    torch.testing.assert_close(via_metric, expected)


def test_one_hundred_distinct_pieces_in_ten_dimensions():
    # Every (input coordinate, output cotangent) pair has its own rank-one
    # piece, despite both matrix dimensions being only 10. A tied rotation
    # cannot express these 100 distinct codes. This tests capacity, not toy recovery.
    reads = torch.eye(10).repeat_interleave(10, dim=1)
    writes = torch.eye(10).repeat(1, 10)
    p = core.FreePieces(reads, writes)
    w = torch.ones(10, 10)
    assert core.reconstruction_error(w, p) == 0
    a = core.free_codes(reads.T, writes.T, p)
    torch.testing.assert_close(a, torch.eye(100))
    assert len(a.argmax(dim=1).unique()) == 100


@pytest.mark.parametrize("shape", [(2, 4), (4, 2)])
def test_projection_gradient_includes_the_compensating_factor_change(shape):
    torch.manual_seed(23)
    w = torch.randn(*shape, dtype=torch.float64)
    p = core.initialize_pieces(w, 7)
    p.read.requires_grad_()
    p.write.requires_grad_()
    h = torch.randn(6, shape[1], dtype=torch.float64)
    g = torch.randn(6, shape[0], dtype=torch.float64)

    def constrained_codes(read, write):
        projected = core.projected_pieces(w, core.FreePieces(read, write))
        return core.free_codes(h, g, projected)

    assert torch.autograd.gradcheck(constrained_codes, (p.read, p.write))
    # With k=0 and scalar cotangents, this is a smooth constrained objective
    # away from zero codes. A sufficiently small projected gradient step must
    # be a descent step on the feasible manifold.
    before = constrained_codes(p.read, p.write).square().sum()
    grad = torch.autograd.grad(before, (p.read, p.write))
    candidate = core.FreePieces(p.read - 1e-6 * grad[0], p.write - 1e-6 * grad[1])
    core.project_piece_sum_(w, candidate)
    after = core.free_codes(h, g, candidate).square().sum()
    assert after < before
    assert core.reconstruction_error(w, candidate) < 1e-12
