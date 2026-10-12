import torch
import pytest
import parauto_core as core
import parauto_method as method
import parauto_structured as structured
import parauto_toy as toy
from parauto_target import ResidualToy, ToyConfig


@pytest.mark.parametrize("precision,dtype", [("fp16", torch.float16), ("bf16", torch.bfloat16)])
def test_mixed_joint_fit_preserves_fp32_factors_and_exact_sum(monkeypatch, precision, dtype):
    if not torch.cuda.is_available():
        pytest.skip('Requires a CUDA allocation')
    torch.manual_seed(85)
    target = ResidualToy(ToyConfig(n_features=7, d_embed=19, d_mlp=3)).cuda().requires_grad_(False)
    cache = structured.cache_events(target, torch.randn(29, 7, device='cuda'))
    named = toy.matrices(target)
    weights = [w for _, w in named]
    entries = [cache['matrices'][name] for name, _ in named]
    initial = [core.initialize_pieces(w, 23) for w in weights]
    cfg = method.TrainConfig(steps=12, batch=11, normalization_events=17, log_interval=1, precision=precision, projection_precision='fp32')
    seen = []
    original = structured.norm_input

    def observe(value):
        if torch.is_autocast_enabled('cuda'):
            seen.append(value.dtype)
        return original(value)

    monkeypatch.setattr(structured, 'norm_input', observe)
    fitted, histories = method.fit(entries, weights, initial, cfg, log=lambda row: None)
    assert dtype in seen
    assert histories[0][-1].get('overflow_skipped_updates', 0) < cfg.steps
    assert any(not torch.equal(a.read, b.read) for a, b in zip(initial, fitted))
    for w, p, rows in zip(weights, fitted, histories):
        assert p.read.dtype == p.write.dtype == torch.float32
        assert torch.isfinite(p.read).all() and torch.isfinite(p.write).all()
        assert core.reconstruction_error(w, p) < 1e-5
        assert rows[-1]['optimizer_updates'] == cfg.steps-rows[-1].get('overflow_skipped_updates', 0)
