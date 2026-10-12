"""joint_residuals equals the brute-force signed sum; gradients reach both factors."""
import torch
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import core, method
from fisher_attribution import capture_fisher


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = target(); data, _ = tokens('val'); rows = batch(data, 0)[:1]
    cache = capture_fisher(model, rows, k=3, seed=1)
    names = ['h.0.attn.q_proj', 'h.0.mlp.c_fc', 'h.3.mlp.down_proj']
    counts = dict(q_proj=1024, c_fc=3072, down_proj=3584)
    index = torch.arange(0, 512, 16, device='cuda')
    entries = [method.slice_entry(cache[MODULES.index(n)], index) for n in names]
    pieces = [core.initialize_pieces(model.get_submodule(n).weight, counts[n.split('.')[-1]], 0) for n in names]
    for p in pieces: p.read.requires_grad_(); p.write.requires_grad_()
    cfg = method.TrainConfig(precision='fp32', joint_fractions=(.01, .1))
    parts = [method.joint_parts(e, p, cfg) for e, p in zip(entries, pieces)]
    values = method.joint_residuals(parts, cfg)
    # brute force in fp64
    with torch.no_grad():
        z = [(e['h'].double() @ p.read.double())[:, None, :]*(e['events']['cot'].double() @ p.write.double()) for e, p in zip(entries, pieces)]
        score = torch.cat([zz.square().mean(1).sqrt() for zz in z], -1)
        for f, v in zip(cfg.joint_fractions, values):
            keep = core.topk_mask(score, max(1, round(f*score.shape[-1])))
            removed = sum(((~m)[:, None, :]*zz).sum(-1) for zz, m in zip(z, keep.split([zz.shape[-1] for zz in z], -1)))
            expected = removed.square().mean()
            assert abs(float(v)-float(expected))/float(expected) < 1e-3, (f, float(v), float(expected))
    values.sum().backward()
    assert all(torch.isfinite(p.read.grad).all() and torch.isfinite(p.write.grad).all() and p.read.grad.abs().sum() > 0 for p in pieces)
    save_json(dict(passed=True, values=[float(v) for v in values]), ROOT/'output/fisher_k4_joint_5000/validation.json')
    print('JOINT_TEST_PASSED', flush=True)


if __name__ == '__main__': main()
