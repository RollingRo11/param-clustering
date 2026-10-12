"""Checks for the usage-density term (VPD's ImportanceMinimalityLoss beta shape)."""
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'par-auto'))    # cluster layout: code/{compare,par-auto}
sys.path.insert(0, str(HERE.parent))               # local layout: par-auto/{runpod_compare,*.py}
import parauto_core as core
import parauto_method as method


def entry_of(h, g):
    return dict(structured=True, kind='scalar_loss', h=h, events={'g': g}, shared={},
                net_rms=torch.zeros(len(h), dtype=h.dtype, device=h.device))


def setup(seed=0, n=64, d_in=12, d_out=10, count=16):
    torch.manual_seed(seed)
    weight = torch.randn(d_out, d_in, dtype=torch.float64)
    pieces = core.initialize_pieces(weight, count, seed)
    return weight, pieces, entry_of(torch.randn(n, d_in, dtype=torch.float64), torch.randn(n, d_out, dtype=torch.float64))


def main():
    torch.manual_seed(0)
    scores = torch.rand(128, 20, dtype=torch.float64)
    off = method.TrainConfig(batch=128, precision='fp32')
    on = method.TrainConfig(batch=128, precision='fp32', density_beta=.5)
    p = .641

    a = method.minimality(scores, off, p)
    b = method.concentration(scores, off, p).mean()
    assert a.equal(b), (float(a), float(b))
    print('density_beta=0 is bitwise the previous objective')

    got = float(method.minimality(scores, on, p))
    power = (scores.abs() + on.lp_epsilon).pow(p)
    want = float(((power.mean(0) - on.lp_epsilon**p) * (1 + .5*torch.log2(1 + power.sum(0)))).sum())
    assert math.isclose(got, want, rel_tol=1e-12), (got, want)
    print(f'matches VPD\'s formula written out by hand ({got:.6g})')

    # What the term is a function of: each piece's MEAN activity over the batch, not how that
    # activity is spread across tokens. Equal mean, different spread, same price.
    spread = torch.full((128, 1), .1, dtype=torch.float64)
    concentrated = torch.zeros(128, 1, dtype=torch.float64); concentrated[:13] = 128*.1/13
    cheap, dear = float(method.minimality(spread, on, 1.)), float(method.minimality(concentrated, on, 1.))
    assert math.isclose(cheap, dear, rel_tol=1e-9), (cheap, dear)
    print(f'equal mean activity, different spread: same price ({cheap:.6g} vs {dear:.6g})')

    # Super-linearity in that mean: twice the activity costs MORE than twice the price.
    one = torch.full((128, 1), .1, dtype=torch.float64)
    two = torch.full((128, 1), .2, dtype=torch.float64)
    single, double = float(method.minimality(one, on, 1.)), float(method.minimality(two, on, 1.))
    assert double > 2*single, (single, double)
    print(f'twice the activity costs {double/single:.3f}x, not 2x (plain Lp at p=1 would be exactly 2x)')

    # So, at the same per-token size, a piece active on more tokens pays more PER USE.
    rare = torch.zeros(128, 1, dtype=torch.float64); rare[:8] = .5
    common = torch.zeros(128, 1, dtype=torch.float64); common[:64] = .5
    per_use = [float(method.minimality(v, on, 1.))/int((v > 0).sum()) for v in (rare, common)]
    assert per_use[1] > per_use[0], per_use
    print(f'price per use: rare piece {per_use[0]:.4g} < frequent piece {per_use[1]:.4g}')

    # Scale invariance of the pricing: dividing scores by a per-matrix scale is what makes the log comparable.
    scale = scores.abs().mean()
    scaled = float(method.minimality(scores, on, p, scale))
    assert scaled != got and math.isfinite(scaled)
    print(f'with a per-matrix scale the value changes as expected ({scaled:.6g})')

    entries, weights, initials = [], [], []
    for i in range(3):
        w, pieces, e = setup(seed=i)
        entries.append(e); weights.append(w); initials.append(pieces)
    run = method.TrainConfig(steps=8, batch=32, precision='fp32', projection_precision='fp64',
                             log_interval=100, normalization_events=32, density_beta=.5)
    fitted, histories = method.fit(entries, weights, initials, run, log=lambda row: None)
    rows = [r for h in histories for r in h]
    assert rows and all(math.isfinite(r['loss']) for r in rows)
    for w, item in zip(weights, fitted):
        assert core.reconstruction_error(w, item) < 1e-9
    print('fit with the density term: finite losses, exact matrix sums kept')

    saved = {}
    method.fit(entries, weights, initials, run, log=lambda row: None,
               checkpoint=lambda state: saved.update(state), checkpoint_interval=4, stop_after=4)
    resumed, _ = method.fit(entries, weights, initials, run, log=lambda row: None, resume=saved, stop_after=4)
    for a_, b_ in zip(fitted, resumed):
        assert torch.allclose(a_.read, b_.read, atol=1e-10) and torch.allclose(a_.write, b_.write, atol=1e-10)
    print('resume at step 4 reproduces the 8-step factors')

    try:
        method.fit(entries, weights, initials,
                   method.TrainConfig(steps=1, batch=8, precision='fp32', density_beta=.5, count_coeff=1.),
                   log=lambda row: None)
    except ValueError:
        print('density with the old activity count is rejected')
    else:
        raise AssertionError('should have been rejected')
    print('CHECK_DENSITY_PASSED')


if __name__ == '__main__':
    main()
