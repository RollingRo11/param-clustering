"""Fisher sensor checks: with true labels and k=1 it reproduces the ordinary cache exactly."""
import torch
from common import ROOT, MODULES, target, tokens, batch, save_json
from train_pair import capture, core, method
from gim_attribution import capture_gim
from fisher_attribution import capture_fisher


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    model = target(); data, _ = tokens('val'); rows = batch(data, 0)[:2]
    base = capture(model, rows)
    same = capture_fisher(model, rows, k=1, labels=True)
    for a, b in zip(base, same):
        torch.testing.assert_close(a['h'], b['h'], rtol=0, atol=0)
        torch.testing.assert_close(a['events']['g'], b['events']['cot'][:, 0], rtol=0, atol=0)
    gim = capture_gim(model, rows); same_gim = capture_fisher(model, rows, k=1, labels=True, gim=True)
    for a, b in zip(gim, same_gim):
        torch.testing.assert_close(a['events']['g'], b['events']['cot'][:, 0], rtol=0, atol=0)
    assert all(not m._forward_hooks for m in model.modules())
    # Sampled version: scores are finite, the k samples differ, and the score equals the RMS formula.
    fisher = capture_fisher(model, rows, k=4, seed=3)
    entry = fisher[MODULES.index('h.2.mlp.c_fc')]
    assert not torch.equal(entry['events']['cot'][:, 0], entry['events']['cot'][:, 1])
    pieces = core.initialize_pieces(model.get_submodule('h.2.mlp.c_fc').weight, 3072, 0)
    small = method.slice_entry(entry, torch.arange(0, 1024, 64, device='cuda'))
    with torch.no_grad():
        score = method.scores_of(small, pieces)
        h, cot = small['h'].float(), small['events']['cot'].float()
        expected = (h @ pieces.read).abs() * ((cot @ pieces.write).square().mean(1)).sqrt()
    torch.testing.assert_close(score, expected, rtol=1e-4, atol=1e-6)
    assert torch.isfinite(score).all()
    save_json(dict(passed=True), ROOT/'output/fisher_rank/validation.json')
    print('FISHER_ATTRIBUTION_TEST_PASSED', flush=True)


if __name__ == '__main__': main()
