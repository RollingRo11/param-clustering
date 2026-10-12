"""Write-size sensor checks on the actual target: cache, score and removal bound."""
import torch
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import capture, core, method
from write_attribution import capture_inputs


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = target()
    data, _ = tokens('val')
    rows = batch(data, 0)[:2]
    # Forward-only inputs match the gradient/GIM cache bit for bit.
    baseline = capture(model, rows)
    cache = capture_inputs(model, rows)
    for a, b in zip(baseline, cache):
        torch.testing.assert_close(a['h'], b['h'], rtol=0, atol=0)
    assert all(not m._forward_hooks for m in model.modules())
    counts = dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)
    worst_score, worst_slack = 0., float('inf')
    for name, entry in zip(MODULES, cache):
        weight = model.get_submodule(name).weight
        pieces = core.initialize_pieces(weight, counts[name.split('.')[-1]], 0)
        index = torch.arange(0, len(entry['h']), 97, device='cuda')
        small = method.slice_entry(entry, index)
        scores = method.scores_of(small, pieces).double()                  # fp32 path, no autocast
        h, r, u = small['h'].double(), pieces.read.double(), pieces.write.double()
        writes = (h @ r).unsqueeze(-1) * u.T.unsqueeze(0)                   # [n, C, d_out]
        exact = writes.norm(dim=-1)
        # FP32 dot products near zero have large relative error; compare against the row's largest score.
        worst_score = max(worst_score, float(((scores-exact).abs()/exact.amax(-1, keepdim=True)).max()))
        # All pieces together give W h exactly; any removed set obeys the triangle bound.
        full = h @ weight.double().T
        assert float((writes.sum(1)-full).norm()/full.norm()) < 1e-5, name
        removed = torch.rand(exact.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(1)) < .3
        change = (writes*removed.unsqueeze(-1)).sum(1).norm(dim=-1)
        bound = (exact*removed).sum(-1)
        worst_slack = min(worst_slack, float((bound-change).min()))
    assert worst_score < 1e-4, worst_score
    assert worst_slack >= -1e-9, worst_slack
    save_json(dict(passed=True, cache_bitwise_equal=True, score_max_relative_error=worst_score,
                   min_bound_slack=worst_slack), ROOT/'output/write_5000/validation.json')
    print('WRITE_ATTRIBUTION_TEST_PASSED', flush=True)


if __name__ == '__main__': main()
