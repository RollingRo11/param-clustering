"""Analytic VJP checks and actual-target forward/capture restoration checks."""
import json
import torch
import torch.nn.functional as F
from common import ROOT, MODULES, target, tokens, batch, save_json
from gim_attribution import TemperatureSoftmax, gim_context, capture_gim
from train_pair import capture


def main():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 5, device='cuda', dtype=torch.float64, requires_grad=True)
    g = torch.randn_like(x)
    x = x.masked_fill(torch.arange(5, device='cuda') == 4, -float('inf'))
    for temperature in (1., 2.):
        y = TemperatureSoftmax.apply(x, -1, temperature)
        torch.testing.assert_close(y, F.softmax(x, dim=-1), rtol=0, atol=0)
        actual = torch.autograd.grad(y, x, g, retain_graph=True)[0]
        p = F.softmax(x/temperature, dim=-1)
        expected = p*(g-(g*p).sum(-1, keepdim=True))
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    model = target()
    data, _ = tokens('val')
    rows = batch(data, 0)[:1]
    original = F.softmax
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(rows[:,:-1])[0]
        with gim_context(model):
            modified = model(rows[:,:-1])[0]
        torch.testing.assert_close(logits, modified, rtol=0, atol=0)
    # Neutral rules reproduce the ordinary cache, including captured Q/K/V.
    baseline = capture(model, rows)
    neutral = capture_gim(model, rows, temperature=1., q_scale=1., k_scale=1., v_scale=1., freeze_rms=False)
    for a,b in zip(baseline, neutral):
        torch.testing.assert_close(a['h'], b['h'], rtol=0, atol=0)
        torch.testing.assert_close(a['events']['g'], b['events']['g'], rtol=0, atol=0)
    # On the final block, changing one branch scales only its local VJP.
    for suffix, scale in [('q_proj', .25), ('k_proj', .25), ('v_proj', .5)]:
        kwargs = dict(temperature=1., q_scale=1., k_scale=1., v_scale=1., freeze_rms=False)
        kwargs[suffix[0]+'_scale'] = scale
        scaled = capture_gim(model, rows, **kwargs)
        i = MODULES.index('h.3.attn.'+suffix)
        torch.testing.assert_close(scaled[i]['events']['g'], neutral[i]['events']['g']*scale,
                                   rtol=0, atol=0)
    gim = capture_gim(model, rows)
    changes = {}
    for name,a,b in zip(MODULES, baseline, gim):
        torch.testing.assert_close(a['h'], b['h'], rtol=0, atol=0)
        assert torch.isfinite(b['events']['g']).all(), name
        changes[name] = float((a['events']['g'].float()-b['events']['g'].float()).norm()
                               / a['events']['g'].float().norm().clamp_min(1e-20))
    assert max(changes.values()) > .1
    try:
        with gim_context(model): raise RuntimeError('restoration test')
    except RuntimeError: pass
    assert F.softmax is original
    assert all(not m._forward_hooks for m in model.modules())
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(model(rows[:,:-1])[0], logits, rtol=0, atol=0)
    save_json(dict(passed=True, forward_bitwise_equal=True, gradient_relative_changes=changes),
              ROOT/'output/gim_1h/validation.json')
    print('GIM_ATTRIBUTION_TEST_PASSED', flush=True)


if __name__ == '__main__': main()
