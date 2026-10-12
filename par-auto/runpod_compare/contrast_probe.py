"""Run the natural-experiment (contrast) sensor on a checkpoint, or check its algebra offline.

  --self-test   no model, no GPU, no data: synthetic checks of the ridge fit and the score.
  otherwise     load a checkpoint from the SPECS table, run contrast_scores over held-out
                sequences, and write diagnostics plus per-matrix score summary statistics.

The probe only measures the sensor; it does not mask anything or compute a KLKeep curve.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT, MODULES, credentials, target, tokens, save_json, sha
from contrast_attribution import contrast_scores, piece_scores, ridge_fit


def self_test():
    """(a) linear teacher recovery, (b) orthogonal piece scores ~0 and aligned piece scales."""
    torch.manual_seed(0)
    result = {}

    # (a) z = T y exactly. With a tiny ridge the solve must return T; with the sensor's default
    # ridge (1e-3 * mean eigenvalue) the fit is deliberately shrunk, so we report that error too
    # and only require its residual fraction to be ~0.
    pairs, d_out, d_model = 4096, 64, 16
    T = torch.randn(d_model, d_out, dtype=torch.float64)
    dy = torch.randn(pairs, d_out, dtype=torch.float64)
    dz = dy @ T.T
    X, fit = ridge_fit(dy.float(), dz.float(), ridge=1e-6, holdout=5)
    exact_error = float((X.T.double() - T).abs().max())
    _, loose = ridge_fit(dy.float(), dz.float(), ridge=1e-3, holdout=5)
    result['teacher'] = dict(pairs=pairs, d_out=d_out, d_model=d_model, small_ridge=1e-6,
                             max_abs_error_small_ridge=exact_error,
                             residual_fraction_small_ridge=fit['residual_fraction'],
                             heldout_residual_fraction_small_ridge=fit['heldout_residual_fraction'],
                             residual_fraction_default_ridge=loose['residual_fraction'])
    assert exact_error < 1e-3, exact_error
    assert fit['residual_fraction'] < 1e-8, fit['residual_fraction']
    assert loose['residual_fraction'] < 1e-4, loose['residual_fraction']

    # (b) With X = I the map M is the identity, so a piece's write difference is delta_rho_c u_c.
    # Piece 0 writes along the contrast direction, piece 1 orthogonal to it, piece 2 along it
    # with twice the magnitude. Reads are chosen so every piece has the same delta_rho.
    d, n, C = 8, 5, 3
    X = torch.eye(d, dtype=torch.float32)
    dz = torch.zeros(n, d); dz[:, 0] = torch.linspace(1., 3., n)          # contrast along e0
    dh = torch.zeros(n, d); dh[:, 3] = torch.linspace(.5, 2.5, n)         # read direction e3
    read = torch.zeros(d, C); read[3, :] = 1.                             # delta_rho_c = dh[:, 3]
    write = torch.zeros(d, C)
    write[0, 0] = 1.                                                      # aligned, magnitude 1
    write[1, 1] = 1.                                                      # orthogonal
    write[0, 2] = 2.                                                      # aligned, magnitude 2
    s = piece_scores(X, read, write, dh, dz)
    expected = dh[:, 3].abs()
    aligned = float((s[:, 0] - expected).abs().max())
    orthogonal = float(s[:, 1].abs().max())
    ratio = float((s[:, 2]/s[:, 0].clamp_min(1e-30)).mean())
    result['pieces'] = dict(aligned_max_abs_error=aligned, orthogonal_max_score=orthogonal,
                            double_magnitude_ratio=ratio,
                            note='score is |r_c.dh| * |<M u_c, dz/||dz||>|; contrast magnitude divides out')
    assert aligned < 1e-5, aligned
    assert orthogonal < 1e-6, orthogonal
    assert abs(ratio - 2.) < 1e-5, ratio

    # (c) pairing respects exclusion and the contrast threshold.
    from contrast_attribution import choose_pairs
    from common import CONTEXT
    m = 2*CONTEXT
    h = torch.randn(m, 6); z = torch.randn(m, 4)
    h[1] = h[0] + 1e-6                       # nearest neighbour of token 0 is its own successor
    sequence = torch.arange(m)//CONTEXT; position = torch.arange(m) % CONTEXT
    candidate = torch.arange(m)
    threshold = float(torch.cdist(z, z).flatten().median())
    partner, cosine, contrast, fell_back = choose_pairs(h, z, candidate, sequence, position,
                                                        threshold, chunk=128)
    same = sequence[partner] == sequence
    adjacent = int(((position[partner]-position).abs() <= 1)[same].sum())
    kept = ~fell_back
    result['pairing'] = dict(tokens=m, adjacent_or_self_pairs=adjacent,
                             fallback_fraction=float(fell_back.float().mean()),
                             min_contrast_non_fallback=float(contrast[kept].min()) if bool(kept.any()) else None,
                             threshold=threshold, mean_cosine=float(cosine.mean()))
    assert adjacent == 0, adjacent
    assert not bool(kept.any()) or float(contrast[kept].min()) > threshold
    result['passed'] = True
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--name', default='gim_5000', help='key in local_sufficiency_probe.SPECS')
    ap.add_argument('--sequences', type=int, default=16)
    ap.add_argument('--first-row', type=int, default=472, help='held-out val rows, as in the other probes')
    ap.add_argument('--microbatch', type=int, default=2)
    ap.add_argument('--candidates', type=int, default=4096)
    ap.add_argument('--exclude-window', type=int, default=1)
    ap.add_argument('--ridge', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    if args.self_test:
        result = self_test()
        result.update(mode='self-test', torch=torch.__version__, source_sha256=sha(__file__),
                      sensor_sha256=sha(Path(__file__).with_name('contrast_attribution.py')),
                      seconds=round(time.perf_counter()-started, 2))
        save_json(result, args.output)
        print(json.dumps(result, indent=1), flush=True)
        print('CONTRAST_SELF_TEST_PASSED', flush=True)
        return

    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    from local_sufficiency_probe import SPECS, load
    if args.name not in SPECS:
        raise ValueError(f'unknown checkpoint {args.name}; known: {sorted(SPECS)}')
    model = target()
    pieces, trainer, meta = load(model, args.name)
    if meta['kind'] != 'parauto':
        raise ValueError('v1 scores free rank-one pieces; VPD has its own CI sensor')
    del trainer
    data, manifest = tokens('val')
    rows = torch.tensor(np.array(data[args.first_row:args.first_row+args.sequences], dtype=np.int64),
                        device='cuda')
    scores, diagnostics = contrast_scores(model, rows, pieces, microbatch=args.microbatch,
                                          candidates=args.candidates, exclude_window=args.exclude_window,
                                          ridge=args.ridge, seed=args.seed)
    generator = torch.Generator(device='cuda').manual_seed(0)
    summary = {}
    for name, s in zip(MODULES, scores):
        flat = s.flatten()
        # torch.quantile refuses very large inputs, so the median/p99 come from a 1M subsample.
        index = torch.randint(0, flat.numel(), (min(flat.numel(), 1_000_000),),
                              generator=generator, device=flat.device)
        sub = flat[index]
        top = s.topk(min(16, s.shape[-1]), dim=-1).values
        summary[name] = dict(components=int(s.shape[-1]), mean=float(flat.mean()),
                             median=float(sub.median()), p99=float(sub.quantile(.99)),
                             max=float(flat.max()), zero_fraction=float((flat == 0).float().mean()),
                             top16_share=float(top.sum()/flat.sum().clamp_min(1e-30)))
        del index, sub, top
    total = torch.cat([s for s in scores], -1)
    counts = [int(s.shape[-1]) for s in scores]
    offsets = [int(v) for v in np.cumsum([0]+counts)]
    k = 193
    head = total.topk(k, dim=-1).indices
    del total
    share = {}
    for name, lo, hi in zip(MODULES, offsets[:-1], offsets[1:]):
        share[name] = float(((head >= lo) & (head < hi)).float().sum()/(len(head)*k))
    result = dict(mode='probe', name=args.name, sequences=args.sequences, first_row=args.first_row,
                  provenance=meta, data_sha256=manifest['val']['sha256'],
                  diagnostics=diagnostics, per_matrix=summary,
                  top193_matrix_share=share, total_components=int(sum(counts)),
                  gpu=torch.cuda.get_device_name(), seconds=round(time.perf_counter()-started, 1),
                  source_sha256=sha(__file__),
                  sensor_sha256=sha(Path(__file__).with_name('contrast_attribution.py')),
                  note=('scores are |projection of a piece write difference onto the real output '
                        'contrast|, in residual-stream norm units; no masking is performed here'))
    save_json(result, args.output)
    print(json.dumps(dict(name=args.name, mean_cosine=diagnostics['mean_cosine'],
                          mean_contrast=diagnostics['mean_contrast'],
                          mean_residual_fraction=diagnostics['mean_residual_fraction'],
                          seconds=result['seconds']), indent=1), flush=True)
    print('CONTRAST_PROBE_DONE', flush=True)


if __name__ == '__main__':
    main()
