import copy
import numpy as np
import pytest
import torch
import parauto_core as core
import parauto_method as method
import parauto_pile_scalar as scalar
from train_pile_scalar import TokenStream, planned_steps
from test_pile import tiny_model


def test_stream_resume_and_epoch_boundary(tmp_path):
    path = tmp_path/'tokens.bin'
    np.arange(30, dtype=np.uint16).tofile(path)
    stream = TokenStream(path, batch=8, context=4, train_stop=25, seed=7)
    offsets = []
    for step in range(3):
        blocks, ix = stream.blocks(step)
        fresh = TokenStream(path, batch=8, context=4, train_stop=25, seed=7)
        expected, expected_ix = fresh.blocks(step)
        assert ix == expected_ix
        torch.testing.assert_close(blocks, expected)
        assert blocks.shape == (2, 5)
        offsets.extend(ix)
    assert sorted(offsets[:5]) == [0, 5, 10, 15, 20]
    assert all(i+5 <= 25 for i in offsets)


def test_stream_batch_is_used_once_and_clean_stop_resumes():
    torch.manual_seed(17)
    w = torch.randn(3, 4)
    initial = [core.initialize_pieces(w, 6)]
    entries = [dict(structured=True, kind='scalar_loss', h=torch.randn(5, 4),
                    events={'g': torch.randn(5, 3)}, shared={}, net_rms=torch.zeros(5))]
    cfg = method.TrainConfig(steps=5, batch=5, precision='fp32', projection_precision='fp32', log_interval=1)
    seen = []
    def provider(step):
        seen.append(step)
        return [dict(entries[0], h=entries[0]['h']*(step+1))]
    expected, _ = method.fit(entries, [w], initial, cfg, batch_provider=provider, log=lambda _: None)
    assert seen == list(range(5))
    seen.clear()
    states = []
    method.fit(entries, [w], initial, cfg, batch_provider=provider, log=lambda _: None,
               step_callback=lambda completed: completed == 2,
               checkpoint=lambda state: states.append(copy.deepcopy(state)), checkpoint_interval=100)
    assert seen == [0, 1] and states[-1]['next_step'] == 2
    actual, _ = method.fit(entries, [w], initial, cfg, resume=states[-1], batch_provider=provider, log=lambda _: None)
    assert seen == list(range(5))
    torch.testing.assert_close(actual[0].read, expected[0].read, atol=0, rtol=0)
    torch.testing.assert_close(actual[0].write, expected[0].write, atol=0, rtol=0)


def test_scalar_bf16_cache_is_rounded_fp32_derivative():
    model = tiny_model()
    ids = torch.tensor([[1, 3, 2, 7]])
    labels = torch.tensor([[3, 2, 7, 1]])
    rounded, ce = scalar.capture(model, ids, labels)
    exact, expected_ce = scalar.capture(model, ids, labels, torch.float32)
    assert ce == expected_ce
    for a, b in zip(rounded, exact, strict=True):
        assert a['h'].dtype == a['events']['g'].dtype == torch.bfloat16
        torch.testing.assert_close(a['h'], b['h'].bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(a['events']['g'], b['events']['g'].bfloat16(), atol=0, rtol=0)


def test_budget_reserves_time_and_rounds_down():
    assert planned_steps(1., hours=8) == 25300
    with pytest.raises(ValueError): planned_steps(0.)
