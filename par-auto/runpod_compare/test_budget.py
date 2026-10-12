import sys
from pathlib import Path
import copy
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'par-auto'))
import parauto_core as core
import parauto_method as method
from common import POSITIONS, CONTEXT, BATCH, STEPS, batch


def test_exact_budget_and_last_partial_batch():
    assert STEPS == 30518
    rows = POSITIONS//CONTEXT
    assert rows == (STEPS-1)*BATCH+37
    assert rows*CONTEXT == POSITIONS
    data = np.zeros((rows, 1), dtype=np.uint16)
    assert batch(data, STEPS-1, 'cpu').shape == (37, 1)


def test_partial_batch_resume_preserves_update():
    torch.manual_seed(7)
    w = torch.randn(3, 4)
    initial = [core.initialize_pieces(w, 6)]
    entries = [dict(structured=True, kind='scalar_loss', h=torch.randn(5, 4),
                    events={'g': torch.randn(5, 3)}, shared={}, net_rms=torch.zeros(5))]
    cfg = method.TrainConfig(steps=2, batch=5, projection_precision='fp32')
    provider = lambda step: [method.slice_entry(entries[0], slice(0, 5 if step == 0 else 3))]
    kwargs = dict(batch_provider=provider, log=lambda _: None, allow_partial_batch=True)
    expected, _ = method.fit(entries, [w], initial, cfg, **kwargs)
    saved = []
    method.fit(entries, [w], initial, cfg, stop_after=1,
               checkpoint=lambda state: saved.append(copy.deepcopy(state)), **kwargs)
    actual, _ = method.fit(entries, [w], initial, cfg, resume=saved[0], **kwargs)
    torch.testing.assert_close(actual[0].read, expected[0].read, atol=0, rtol=0)
    torch.testing.assert_close(actual[0].write, expected[0].write, atol=0, rtol=0)
    assert core.reconstruction_error(w, actual[0]) < 1e-5
    with pytest.raises(ValueError, match='exactly'):
        method.fit(entries, [w], initial, cfg, batch_provider=provider, log=lambda _: None)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_bounded_kl_matches_upstream_value_and_gradient(dtype):
    from train_pair import bounded_recon_kl
    from param_decomp.models.batch_and_loss_fns import recon_loss_kl
    torch.manual_seed(10)
    pred = torch.randn(3, 1001, 33, device='cuda', dtype=dtype, requires_grad=True)
    target = torch.randn_like(pred)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        expected, n = recon_loss_kl(pred, target)
        actual, actual_n = bounded_recon_kl(pred, target)
    expected_grad, = torch.autograd.grad(expected, pred)
    actual_grad, = torch.autograd.grad(actual, pred)
    assert n == actual_n
    torch.testing.assert_close(actual, expected.float(), rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_ci_block_checkpoint_preserves_gradients():
    from param_decomp.models.components import TransformerBlock
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(9)
    block = TransformerBlock(32, 4, [64], max_len=12).cuda()
    x = torch.randn(2, 12, 32, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        direct = block(x)
        recomputed = checkpoint(block, x, use_reentrant=False)
    variables = [x, *block.parameters()]
    a = torch.autograd.grad(direct.square().sum(), variables)
    b = torch.autograd.grad(recomputed.square().sum(), variables)
    torch.testing.assert_close(direct, recomputed, atol=0, rtol=0)
    for first, second in zip(a, b): torch.testing.assert_close(first, second, atol=0, rtol=0)
