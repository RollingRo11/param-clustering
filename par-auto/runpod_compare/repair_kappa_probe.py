"""Run the self-repair correlation kappa on held-out validation sequences and report it.

kappa_c = corr over tokens between a piece's activation rho_c = r_c . h and the projection of
everything written into the residual stream after that piece's site onto the piece's own write
direction u_c (see repair_correlation.py for the definition and for why only o_proj / down_proj
qualify). Very negative kappa means the later layers react against the piece: removing it would be
repaired. A kappa near zero with a large mean projection is a cancellation that is always there,
not a reaction.

The JSON also carries, for each matrix, the correlation across pieces between kappa_c and the
piece's mean Fisher K=8 score -- i.e. whether the pieces the Fisher sensor likes are the ones the
model repairs.

  --self-test  runs the streaming estimator against synthetic series with a closed-form correlation
               and against itself at different batch sizes. No model, no GPU, no cluster.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from repair_correlation import RESIDUAL_SITES, PairedMoments, repair_correlations

SPECS = {
    'vpd_published': ('vpd', 400000, 'checkpoints/published_vs_finished/vpd.pth'),
    'vpd_5000': ('vpd', 5000, 'checkpoints/step_5000/vpd.pth'),
    'gim_5000': ('parauto', 5000, 'output/gim_1h/checkpoints/005000/model.pth'),
    'gim_7783': ('parauto', 7783, 'output/gim_1h/checkpoints/007783/model.pth'),
    'fisher_k4_5000': ('parauto', 5000, 'output/fisher_k4_5000/checkpoints/005000/model.pth'),
    'local_avg_5000': ('parauto', 5000, 'output/local_avg_c1_5000/checkpoints/005000/model.pth'),
    'density_b0.5_5000': ('parauto', 5000, 'output/density_b0.5_5000/checkpoints/005000/model.pth'),
}


def load(model, name):
    """Same convention as local_sufficiency_probe: pieces aligned with common.MODULES."""
    from train_pair import core
    kind, step, rel = SPECS[name]
    path = ROOT/rel
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['method'] == kind and saved['completed'] == step and saved['target_sha256'] == TARGET_SHA
    if kind == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True)
        handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors'])
        trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(),
                                  trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda().float(), p['write'].cuda().float()) for p in saved['factors']]
    return pieces, dict(checkpoint=str(path), checkpoint_sha256=sha(path), kind=kind, step=step)


def finite(value):
    """JSON cannot carry NaN (save_json uses allow_nan=False); an undefined kappa stays undefined."""
    value = float(value)
    return value if np.isfinite(value) else None


def summarize(kappa):
    good = kappa[torch.isfinite(kappa)]
    if good.numel() == 0:
        return {'pieces': int(kappa.numel()), 'defined': 0, 'mean': None, 'median': None,
                'p10': None, 'p90': None, 'fraction_below_-0.3': None}
    q = good.quantile(torch.tensor([.1, .5, .9], dtype=good.dtype, device=good.device))
    return {'pieces': int(kappa.numel()), 'defined': int(good.numel()), 'mean': finite(good.mean()),
            'median': finite(q[1]), 'p10': finite(q[0]), 'p90': finite(q[2]),
            'fraction_below_-0.3': float((good < -.3).double().mean())}


def fisher_means(model, pieces, data, first_row, sequences, k, sites):
    """Mean Fisher K=8 score per piece over the same held-out rows, for the residual-write sites."""
    from fisher_attribution import capture_fisher
    from train_pair import method
    index = {n: i for i, n in enumerate(MODULES)}
    total = {s: torch.zeros(pieces[index[s]].count, dtype=torch.float64, device='cuda') for s in sites}
    n_tok = 0
    for first in range(first_row, first_row+sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda')
        cache = capture_fisher(model, rows, k=k, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for s in sites:
                i = index[s]
                scores = method.scores_of(cache[i], pieces[i]).float()                  # [tokens, C]
                total[s] += scores.double().sum(0)
                del scores
        n_tok += rows.shape[0]*(rows.shape[1]-1)
        del cache, rows
    return {s: (v/n_tok) for s, v in total.items()}, n_tok


def across_pieces(kappa, score):
    """Pearson correlation across the pieces of one matrix, over pieces where kappa is defined."""
    ok = torch.isfinite(kappa) & torch.isfinite(score)
    if int(ok.sum()) < 3:
        return None
    a, b = kappa[ok].double(), score[ok].double()
    if a.std() == 0 or b.std() == 0:
        return None
    return finite(torch.corrcoef(torch.stack([a, b]))[0, 1])


def self_test():
    """The streaming estimator, with no model: a known correlation, and batching invariance."""
    torch.manual_seed(0)
    n, c = 5000, 37
    x = torch.randn(n, c, dtype=torch.float64)
    z = torch.randn(n, c, dtype=torch.float64)
    x = x - x.mean(0)
    z = z - z.mean(0)
    z = z - (z*x).sum(0)/(x*x).sum(0)*x                       # exactly orthogonal to x, column-wise
    a = torch.linspace(-2., 2., c, dtype=torch.float64)
    b = torch.linspace(.25, 3., c, dtype=torch.float64)
    y = a*x + b*z + 17.5                                      # a big constant offset, on purpose
    # corr(x, a x + b z) with x . z = 0 and both centred:  a|x| / sqrt(a^2|x|^2 + b^2|z|^2)
    xn, zn = x.norm(dim=0), z.norm(dim=0)
    known = (a*xn)/torch.sqrt((a*xn).square() + (b*zn).square())

    one = PairedMoments(c).update(x, y).result()
    error = float((one['kappa']-known).abs().max())
    assert error < 1e-6, f'streaming correlation off by {error}'

    five = PairedMoments(c)
    for chunk in range(5):
        lo, hi = chunk*n//5, (chunk+1)*n//5
        five.update(x[lo:hi], y[lo:hi])
    five = five.result()
    batching = float((five['kappa']-one['kappa']).abs().max())
    assert batching < 1e-6, f'batching changes kappa by {batching}'

    # The bookkeeping must also be right, or a constant cancellation cannot be spotted.
    mean_error = float((five['projection_mean']-y.mean(0)).abs().max())
    std_error = float((five['projection_std']-y.std(0, unbiased=False)).abs().max())
    assert mean_error < 1e-6 and std_error < 1e-6, (mean_error, std_error)

    # A perfectly anti-correlated series must come back as exactly -1.
    flat = PairedMoments(c).update(x, -3.*x + 4.).result()
    assert float((flat['kappa']+1.).abs().max()) < 1e-6, float(flat['kappa'].max())

    # A constant later-write (no reaction at all) has no correlation to report, not a zero.
    dead = PairedMoments(c).update(x, torch.full_like(x, 2.)).result()
    assert bool(torch.isnan(dead['kappa']).all())

    print(json.dumps(dict(known_correlation_max_error=error, batching_max_difference=batching,
                          projection_mean_max_error=mean_error, projection_std_max_error=std_error,
                          tokens=n, pieces=c)), flush=True)
    print('REPAIR_KAPPA_SELF_TEST_PASSED', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--name', choices=sorted(SPECS))
    ap.add_argument('--sequences', type=int, default=64)
    ap.add_argument('--fisher-sequences', type=int, default=8)
    ap.add_argument('--first-row', type=int, default=1024, help='held out from the other probes')
    ap.add_argument('--fisher-k', type=int, default=8)
    ap.add_argument('--top', type=int, default=10, help='how many most-negative pieces to list')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    assert args.name and args.output, '--name and --output are required unless --self-test'
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    started = time.perf_counter()
    model = target()
    pieces, meta = load(model, args.name)
    data, manifest = tokens('val')
    assert args.first_row+max(args.sequences, args.fisher_sequences) <= data.shape[0]

    def rows_iter():
        for first in range(args.first_row, args.first_row+args.sequences, 2):
            yield torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda')

    report = lambda batches, tok: print(json.dumps(dict(batches=batches, tokens=tok,
                                                        seconds=round(time.perf_counter()-started, 1))), flush=True)
    out = repair_correlations(model, pieces, rows_iter(), progress=report)
    fisher, fisher_tokens = fisher_means(model, pieces, data, args.first_row, args.fisher_sequences,
                                         args.fisher_k, RESIDUAL_SITES)

    per_matrix = {}
    for name in RESIDUAL_SITES:
        stats = out['sites'][name]
        kappa = stats['kappa']
        score = fisher[name]
        order = torch.argsort(torch.nan_to_num(kappa, nan=float('inf')))[:args.top]
        per_matrix[name] = dict(
            kappa=summarize(kappa),
            rho_mean_abs=finite(stats['rho_mean'].abs().mean()),
            rho_std_mean=finite(stats['rho_std'].mean()),
            projection_mean_abs=finite(stats['projection_mean'].abs().mean()),
            projection_std_mean=finite(stats['projection_std'].mean()),
            later_write_norm_mean=stats['later_write_norm_mean'],
            fisher_mean_over_pieces=finite(score.mean()),
            kappa_vs_fisher=across_pieces(kappa, score),
            most_negative=[dict(piece=int(i), kappa=finite(kappa[i]),
                                projection_mean=finite(stats['projection_mean'][i]),
                                projection_std=finite(stats['projection_std'][i]),
                                rho_std=finite(stats['rho_std'][i]),
                                fisher=finite(score[i])) for i in order.tolist()
                           if torch.isfinite(kappa[i])])
    all_kappa = torch.cat([out['sites'][n]['kappa'] for n in RESIDUAL_SITES])
    all_fisher = torch.cat([fisher[n] for n in RESIDUAL_SITES])

    result = dict(name=args.name, provenance=meta, sites=list(RESIDUAL_SITES),
                  sequences=args.sequences, first_row=args.first_row, tokens=out['tokens'],
                  batches=out['batches'], fisher_k=args.fisher_k,
                  fisher_sequences=args.fisher_sequences, fisher_tokens=fisher_tokens,
                  hook_checks=out['checks'], per_matrix=per_matrix,
                  overall=dict(kappa=summarize(all_kappa), kappa_vs_fisher=across_pieces(all_kappa, all_fisher)),
                  gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'],
                  source_sha256=sha(__file__), module_sha256=sha(Path(__file__).with_name('repair_correlation.py')),
                  seconds=round(time.perf_counter()-started, 1),
                  note=('kappa is undefined for h.3.mlp.down_proj: nothing is written after it. '
                        'A near-zero kappa with a large |projection_mean| is a constant cancellation, '
                        'not a reaction to the piece.'))
    save_json(result, args.output)
    for name in RESIDUAL_SITES:
        m = per_matrix[name]
        print(json.dumps({name: dict(kappa_mean=m['kappa']['mean'], kappa_median=m['kappa']['median'],
                                     kappa_p10=m['kappa']['p10'], frac_below_neg0_3=m['kappa']['fraction_below_-0.3'],
                                     proj_mean_abs=None if m['projection_mean_abs'] is None else round(m['projection_mean_abs'], 4),
                                     proj_std_mean=None if m['projection_std_mean'] is None else round(m['projection_std_mean'], 4),
                                     kappa_vs_fisher=m['kappa_vs_fisher'])}), flush=True)
    print(json.dumps(dict(overall=result['overall'], checks=out['checks'])), flush=True)
    print('REPAIR_KAPPA_PROBE_DONE', flush=True)


if __name__ == '__main__':
    main()
