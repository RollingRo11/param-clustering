"""Finite-ablation checks on the actual target.

Each score is compared with brute force: subtract one piece's write at one token inside
the FP32 target, rerun it, and measure the change at the same place the score does.
"""
import copy
import torch
import torch.nn.functional as F
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import capture, core, method
import finite_ablation as fa

COUNTS = dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)


def brute(model, ids, name, piece_write, token):
    """FP64 norm of the change where the score measures it, after removing piece_write at `token`."""
    layer, suffix = name.rsplit('.', 1)
    probe = dict(q_proj='attn.o_proj', k_proj='attn.o_proj', v_proj='attn.o_proj', c_fc='mlp.down_proj')
    block = layer.rsplit('.', 1)[0]
    where = model.get_submodule(block+'.'+probe[suffix]) if suffix in probe else model.get_submodule(name)
    seen = {}
    def grab(_m, inputs, out): seen['x'] = (inputs[0] if suffix in probe else out).detach().clone()
    def remove(_m, _i, out):
        out = out.clone(); out[0, token] -= piece_write; return out
    values = []
    for edit in (False, True):
        # The edit hook is registered first so a probe on the same module sees the edited output.
        hooks = [model.get_submodule(name).register_forward_hook(remove)] if edit else []
        hooks.append(where.register_forward_hook(grab))
        try:
            with torch.no_grad(): model(ids)
        finally:
            for h in hooks: h.remove()
        values.append(seen['x'][0])
    change = values[1]-values[0]                      # [T, d]
    return float(change.norm() if suffix in ('k_proj', 'v_proj') else change[token].norm())


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = target()
    data, _ = tokens('val')
    rows = batch(data, 0)[:2]
    # 1. Forward-only inputs match the gradient/GIM cache bit for bit.
    base = capture(model, rows)
    cached = fa.capture_finite(model, rows)
    for a, b in zip(base, cached):
        torch.testing.assert_close(a['h'], b['h'], rtol=0, atol=0)
    assert all(not m._forward_hooks for m in model.modules())
    # 2. The hand-written GELU backward matches autograd of the plain expression.
    torch.manual_seed(0)
    z, a, u = torch.randn(40, 96, device='cuda'), torch.randn(40, 72, device='cuda', requires_grad=True), torch.randn(96, 72, device='cuda', requires_grad=True)
    g = torch.randn(40, 72, device='cuda')
    ref = (F.gelu(z[:, None], approximate='tanh') - F.gelu(z[:, None]-a[:, :, None]*u.T[None], approximate='tanh')).square().sum(-1)
    ga, gu = torch.autograd.grad((ref*g).sum(), (a, u))
    out = fa.GeluEffect.apply(z, a, u)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    ma, mu = torch.autograd.grad((out*g).sum(), (a, u))
    torch.testing.assert_close(ma, ga, rtol=1e-4, atol=1e-4); torch.testing.assert_close(mu, gu, rtol=1e-4, atol=1e-4)
    # 3. Brute force in FP32 on one sequence: removal at 3 tokens, 3 pieces each.
    ids = rows[:1, :-1]
    exact = copy.deepcopy(model).double()
    entries = fa.capture_finite(model, rows[:1], autocast=False)
    lowp_entries = fa.capture_finite(model, rows[:1])
    tokens_ = torch.tensor([0, 137, 511], device='cuda')
    worst, lowp_worst, report = 0., 0., {}
    for name, entry, lowp in zip(MODULES, entries, lowp_entries):
        if name.split('.')[1] not in ('0', '3'): continue
        weight = model.get_submodule(name).weight
        pieces = core.initialize_pieces(weight, COUNTS[name.split('.')[-1]], 0)
        small = method.slice_entry(entry, tokens_)
        with torch.no_grad():
            score = method.scores_of(small, pieces).double()                       # FP32 path
            # Same BF16-forward cache, scored in FP32 vs with the BF16 training matmuls.
            same = method.slice_entry(lowp, tokens_)
            same_fp32 = method.scores_of(same, pieces).double()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                low = method.scores_of(same, pieces).double()
        errors = []
        for row, token in enumerate(tokens_.tolist()):
            order = score[row].argsort(descending=True)
            for c in (order[0], order[len(order)//2], order[-1]):
                code = entry['h'][token].double() @ pieces.read[:, c].double()
                truth = brute(exact, ids, name, pieces.write[:, c].double()*code, token)
                # FP32 scores have a small absolute noise floor; judge tiny effects against the row maximum.
                scale = max(truth, 1e-3*float(score[row].max()), 1e-12)
                errors.append(abs(float(score[row, c])-truth)/scale)
            top = same_fp32[row] > 1e-2*same_fp32[row].max()
            lowp_row = float(((low[row]-same_fp32[row]).abs()/same_fp32[row].max())[top].max())
            lowp_worst = max(lowp_worst, lowp_row)
        report[name] = max(errors); worst = max(worst, max(errors))
        print(name, entry['effect'], f'max relative error {max(errors):.2e}; bf16 matmul error {lowp_row:.2e}', flush=True)
    assert worst < 1e-2, report
    assert lowp_worst < 5e-2, lowp_worst
    # 4. Training path: gradients reach both factors and are finite.
    entry = method.slice_entry(lowp_entries[MODULES.index('h.0.attn.k_proj')], torch.arange(64, 192, device='cuda'))
    pieces = core.initialize_pieces(model.get_submodule('h.0.attn.k_proj').weight, 1024, 0)
    pieces.read.requires_grad_(); pieces.write.requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        method.scores_of(entry, pieces).sqrt().sum().backward()
    assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in (pieces.read, pieces.write))
    save_json(dict(passed=True, cache_bitwise_equal=True, fp32_max_relative_error=report,
                   bf16_vs_fp32_max_error_top_pieces=lowp_worst), ROOT/'output/finite_5000/validation.json')
    print('FINITE_ABLATION_TEST_PASSED', flush=True)


if __name__ == '__main__': main()
