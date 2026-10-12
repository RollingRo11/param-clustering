"""Checks for the layer-local sufficiency term: exactness, brute force, metric, fit and resume."""
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
                net_rms=torch.zeros(len(h), dtype=torch.float64, device=h.device))


def setup(seed=0, n=64, d_in=12, d_out=10, count=16, device='cpu'):
    torch.manual_seed(seed)
    weight = torch.randn(d_out, d_in, dtype=torch.float64, device=device)
    pieces = core.initialize_pieces(weight, count, seed)
    h = torch.randn(n, d_in, dtype=torch.float64, device=device)
    g = torch.randn(n, d_out, dtype=torch.float64, device=device)
    return weight, pieces, entry_of(h, g)


def brute_force(entry, pieces, weight, mask, nonlinearity):
    """Straightforward loop over the off set, with no algebra shortcuts."""
    h = entry['h']
    off = torch.zeros(len(h), pieces.write.shape[0], dtype=h.dtype, device=h.device)
    for c in range(pieces.read.shape[1]):
        keep = mask[:, c].to(h.dtype)
        off = off + (1 - keep)[:, None] * (h @ pieces.read[:, c])[:, None] * pieces.write[:, c][None, :]
    y = h @ weight.T
    if nonlinearity == 'linear':
        error = off
    else:
        phi = method.LOCAL_NONLINEARITIES[nonlinearity]
        error = phi(y - off) - phi(y)
    return error.square().sum(-1).mean()


def main():
    cfg = method.TrainConfig(precision='fp32', local_coeff=1., local_fractions=(.25,), local_metric='raw')
    weight, pieces, entry = setup()
    count = pieces.read.shape[1]
    full = torch.ones(len(entry['h']), count, dtype=torch.bool, device=entry['h'].device)
    for nonlinearity in ('linear', 'gelu_tanh', 'relu'):
        spec = dict(nonlinearity=nonlinearity, metric=None)
        value = float(method.local_error(entry, pieces, weight, full, spec, cfg))
        assert value < 1e-18, (nonlinearity, value)          # keeping every piece reproduces the matrix exactly
    print('all pieces kept: local error is zero for every nonlinearity')

    torch.manual_seed(1)
    mask = torch.rand(len(entry['h']), count, device=entry['h'].device) > .5
    for nonlinearity in ('linear', 'gelu_tanh', 'relu'):
        spec = dict(nonlinearity=nonlinearity, metric=None)
        got = float(method.local_error(entry, pieces, weight, mask, spec, cfg))
        want = float(brute_force(entry, pieces, weight, mask, nonlinearity))
        assert math.isclose(got, want, rel_tol=1e-9), (nonlinearity, got, want)
        print(f'{nonlinearity}: matches the brute-force off-set loop ({got:.6g})')

    # The metric factor must reproduce e^T M e with M = L L^T.
    torch.manual_seed(2)
    factor = torch.randn(weight.shape[0], 5, dtype=torch.float64)
    spec = dict(nonlinearity='linear', metric=factor)
    got = float(method.local_error(entry, pieces, weight, mask, spec, cfg))
    h = entry['h']
    off = ((h @ pieces.read) * (~mask).double()) @ pieces.write.T
    want = float(((off @ (factor @ factor.T)) * off).sum(-1).mean())
    assert math.isclose(got, want, rel_tol=1e-9), (got, want)
    print(f'metric factor: ||L^T e||^2 equals e^T (L L^T) e ({got:.6g})')

    # Cancellation inside the off set is allowed: two opposite writes cost nothing together.
    twin = core.FreePieces(pieces.read.clone(), pieces.write.clone())
    twin.read[:, 1] = twin.read[:, 0]; twin.write[:, 1] = -twin.write[:, 0]
    pair = torch.ones(len(h), count, dtype=torch.bool); pair[:, :2] = False
    single = pair.clone(); single[:, 1] = True
    together = float(method.local_error(entry, twin, twin.write @ twin.read.T, pair, dict(nonlinearity='linear', metric=None), cfg))
    alone = float(method.local_error(entry, twin, twin.write @ twin.read.T, single, dict(nonlinearity='linear', metric=None), cfg))
    assert together < 1e-18 < alone, (together, alone)
    print(f'cancelling pair: together {together:.3g}, one alone {alone:.6g}')

    # fit() with the term: finite loss, exact sums kept, and resume reproduces the same factors.
    entries, weights, initials, specs = [], [], [], []
    for i in range(3):
        w, p, e = setup(seed=i, device='cpu')
        entries.append(e); weights.append(w); initials.append(p)
        specs.append(dict(nonlinearity='gelu_tanh' if i == 1 else 'linear', metric=None))
    run = method.TrainConfig(steps=8, batch=32, precision='fp32', projection_precision='fp64', log_interval=100,
                             normalization_events=32, local_coeff=.5, local_fractions=(.1, .25, .5), local_metric='raw')
    fitted, histories = method.fit(entries, weights, initials, run, log=lambda row: None, local=specs)
    rows = [r for h in histories for r in h]
    assert all(math.isfinite(r['loss']) for r in rows) and rows
    for w, p in zip(weights, fitted):
        assert core.reconstruction_error(w, p) < 1e-9
    print('fit with the local term: finite losses, exact matrix sums kept')

    saved = {}
    method.fit(entries, weights, initials, run, log=lambda row: None, local=specs,
               checkpoint=lambda state: saved.update(state), checkpoint_interval=4, stop_after=4)
    resumed, _ = method.fit(entries, weights, initials, run, log=lambda row: None, local=specs,
                            resume=saved, stop_after=4)
    for a, b in zip(fitted, resumed):
        assert torch.allclose(a.read, b.read, atol=1e-10) and torch.allclose(a.write, b.write, atol=1e-10)
    print('resume at step 4 reproduces the 8-step factors')

    for bad in (dict(local_coeff=1., local_fractions=()), dict(local_coeff=1., local_fractions=(.1,), local_metric='avg')):
        try:
            method.fit(entries, weights, initials, method.TrainConfig(steps=1, batch=8, precision='fp32', **bad),
                       log=lambda row: None, local=specs)
        except ValueError:
            continue
        raise AssertionError(f'configuration should have been rejected: {bad}')
    print('invalid configurations are rejected')
    print('CHECK_LOCAL_PASSED')


if __name__ == '__main__':
    main()
