"""Why does published VPD not recover when every component that any donor token needs is switched on? In
aggregation67.py's 'all texts' rung (mask 1 on the union U of the needed sets of all 524,288 donor positions, the
recipient's own label elsewhere) the KL stays +0.95 above the own explanation, while every component on gives 0.012.
The two masks differ only on R, the components never needed (label <= 0.1) at any donor position.

This probe measures R. Per matrix: its share of the weight (sum of ||v_c||^2 ||u_c||^2, and the norm of its summed
matrix, both over ||W||_F) and, on the recipients' intact inputs, its share of the matrix output, next to the output
error of the own explanation and of the union-on mask. Then the KL (as in aggregation67.py: KL(P_intact || Q_masked) per
position, mean over a text's 512 positions, then over texts, delta component left out) for masks that switch R on or
off around the union and the own explanation, overall, per layer and per matrix type.
Usage: vpd_union_probe67.py --checkpoint PATH --output JSON [--recipients 1024] [--donors 1024]"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch

from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha

POSITIONS = 512
RECIPIENTS, DONORS = 0, 1024           # first val rows of each, as in aggregation67.py
NEEDED = 0.1


def bootstrap(values, generator, resamples=10000):
    """Mean of a per-text array and the 95% interval of that mean, resampling texts."""
    v = torch.as_tensor(values, dtype=torch.float64)
    means = v[torch.randint(len(v), (resamples, len(v)), generator=generator)].mean(-1)
    return dict(mean=float(v.mean()), lo=float(means.quantile(.025)), hi=float(means.quantile(.975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--recipients', type=int, default=1024); ap.add_argument('--donors', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=16, help='texts per forward')
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    started = time.perf_counter()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'vpd' and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'))
    model = target()
    from train_pair import VPD
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    from param_decomp.models.components import make_mask_infos
    trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
    trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
    del saved
    counts = [trainer.model.module_to_c[n] for n in MODULES]
    total = sum(counts)
    kinds = list(dict.fromkeys(n.split('.', 2)[2] for n in MODULES))
    matrix_of = torch.cat([torch.full((c,), i) for i, c in enumerate(counts)]).cuda()
    layer_of = torch.cat([torch.full((c,), int(n.split('.')[1])) for n, c in zip(MODULES, counts)]).cuda()
    kind_of = torch.cat([torch.full((c,), kinds.index(n.split('.', 2)[2])) for n, c in zip(MODULES, counts)]).cuda()
    data, manifest = tokens('val')
    assert len(data) >= DONORS + args.donors and args.recipients <= DONORS

    def rows_at(first, n):
        return torch.tensor(np.array(data[first:first+n], dtype=np.int64), device='cuda')

    def explain(ids):
        """Labels in [0, 1] [n, 512, total] and the intact inputs of every matrix."""
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            base = trainer.model(ids, cache_type='input')
            ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
            out = torch.cat([ci[n] for n in MODULES], -1).float().clamp(0., 1.)
        return out, base.cache

    def forward(ids, mask):
        """Logits with every matrix replaced by its masked components, delta left out."""
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, mask.split(counts, -1))))).float()

    # 1. Donors: U = needed (label > 0.1) at >= 1 donor position; R = the rest.
    union = torch.zeros(total, dtype=torch.bool, device='cuda')
    label_sum = torch.zeros(total, dtype=torch.float64, device='cuda')
    label_max = torch.zeros(total, device='cuda')
    for first in range(DONORS, DONORS + args.donors, args.chunk):
        n = min(args.chunk, DONORS + args.donors - first)
        e, _ = explain(rows_at(first, n)[:, :-1])
        e = e.flatten(0, 1)
        union |= (e > NEEDED).any(0); label_sum += e.double().sum(0); label_max = torch.maximum(label_max, e.max(0).values)
        del e
    donor_positions = args.donors * POSITIONS
    rest = ~union
    print(dict(donor_positions=donor_positions, total=total, union=int(union.sum()), rest=int(rest.sum()),
               seconds=round(time.perf_counter() - started)), flush=True)

    # 2. Weights: R's share of each matrix.
    weights = []
    for i, name in enumerate(MODULES):
        comp = trainer.model.components[name]
        V, U = comp.V.detach().double(), comp.U.detach().double()          # [d_in, C], [C, d_out]
        W = trainer.model.target_weight(name).detach().double()            # [d_out, d_in]
        assert W.shape == (U.shape[1], V.shape[0]), (name, W.shape, V.shape, U.shape)
        r, u = rest[matrix_of == i], union[matrix_of == i]
        size = V.pow(2).sum(0) * U.pow(2).sum(1)                           # ||v_c||^2 ||u_c||^2
        w2 = float(W.pow(2).sum())
        weights.append(dict(module=name, components=counts[i], union=int(u.sum()), rest=int(r.sum()),
                            mass_all=float(size.sum()) / w2, mass_rest=float(size[r].sum()) / w2,
                            rest_net=float((V[:, r] @ U[r]).norm()) / w2**.5,
                            union_net_error=float(((V[:, u] @ U[u]).T - W).norm()) / w2**.5,
                            delta=float(((V @ U).T - W).norm()) / w2**.5,
                            rest_largest_mass=float(size[r].max()) / w2 if r.any() else 0.))
        del V, U, W

    # 3. Recipients: per matrix output shares at the intact inputs, and the KL of each mask.
    masks = {
        'own': lambda e: e,
        'all_on': lambda e: torch.ones_like(e),
        'union_on': lambda e: torch.where(union, 1., e),                 # aggregation67.py's 'all texts' rung
        'union_on_rest_zero': lambda e: torch.where(union, 1., torch.zeros_like(e)),
        'union_on_rest_half': lambda e: torch.where(union, 1., torch.full_like(e, .5)),
        'own_rest_on': lambda e: torch.where(rest, 1., e),
        'own_rest_zero': lambda e: torch.where(rest, 0., e),
    }
    for L in range(4):
        masks[f'union_on_rest_on_layer{L}'] = lambda e, L=L: torch.where(union | (rest & (layer_of == L)), 1., e)
        masks[f'own_union_on_layer{L}'] = lambda e, L=L: torch.where(union & (layer_of == L), 1., e)
    for k, kind in enumerate(kinds):
        masks[f'union_on_rest_on_{kind}'] = lambda e, k=k: torch.where(union | (rest & (kind_of == k)), 1., e)
    kl = {name: [] for name in masks}
    out_sq = {key: torch.zeros(len(MODULES), dtype=torch.float64) for key in ('target', 'rest', 'own_error', 'union_error', 'all_error')}
    for first in range(RECIPIENTS, RECIPIENTS + args.recipients, args.chunk):
        tick = time.perf_counter()
        n = min(args.chunk, RECIPIENTS + args.recipients - first)
        ids = rows_at(first, n)[:, :-1]
        e, cache = explain(ids)
        with torch.no_grad():
            for i, name in enumerate(MODULES):
                comp = trainer.model.components[name]
                x = cache[name].float()
                acts = x @ comp.V.float()                                    # [n, 512, C]
                y = x @ trainer.model.target_weight(name).float().T
                g, r, u = e[..., matrix_of == i], rest[matrix_of == i], union[matrix_of == i]
                U = comp.U.float()
                for key, value in (('target', y), ('rest', acts[..., r] @ U[r]), ('own_error', (acts * g) @ U - y),
                                   ('union_error', (acts * torch.where(u, 1., g)) @ U - y), ('all_error', acts @ U - y)):
                    out_sq[key][i] += float(value.double().pow(2).sum())
                del x, acts, y
            del cache
            with torch.autocast('cuda', dtype=torch.bfloat16): logp = model(ids)[0].float().log_softmax(-1)
            p = logp.exp()
            for name, make in masks.items():
                q = forward(ids, make(e)).log_softmax(-1)
                kl[name].append((p * (logp - q)).sum(-1).mean(-1).cpu())
                del q
        del e, logp, p
        print(dict(recipients_done=first + n - RECIPIENTS, chunk_seconds=round(time.perf_counter() - tick, 1),
                   own=round(float(torch.cat(kl['own']).mean()), 4), union_on=round(float(torch.cat(kl['union_on']).mean()), 4),
                   all_on=round(float(torch.cat(kl['all_on']).mean()), 4)), flush=True)

    generator = torch.Generator().manual_seed(0)
    kl = {name: torch.cat(v) for name, v in kl.items()}
    results = {name: dict(kl=bootstrap(v, generator), rise_over_own=bootstrap(v - kl['own'], generator)) for name, v in kl.items()}
    for i, w in enumerate(weights):
        t = float(out_sq['target'][i])
        w.update({f'output_{key}': (float(out_sq[key][i]) / t)**.5 for key in ('rest', 'own_error', 'union_error', 'all_error')})
    rest_label_max = label_max[rest]
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta, recipients=[RECIPIENTS, RECIPIENTS + args.recipients],
                  donors=[DONORS, DONORS + args.donors], positions_per_text=POSITIONS, data_sha256=manifest['val']['sha256'],
                  total=total, union=int(union.sum()), rest=int(rest.sum()),
                  rest_label=dict(max_over_donors_max=float(rest_label_max.max()) if rest.any() else None,
                                  dead=int((label_sum[rest] / donor_positions <= 1e-6).sum())),
                  per_layer={L: dict(union=int((union & (layer_of == L)).sum()), rest=int((rest & (layer_of == L)).sum())) for L in range(4)},
                  weights=weights, results=results, gpu=torch.cuda.get_device_name(), seconds=time.perf_counter() - started)
    save_json(result, args.output)
    for name, r in results.items():
        print(f"{name:32s} KL {r['kl']['mean']:.4f}  rise {r['rise_over_own']['mean']:+.4f}", flush=True)
    for w in weights:
        print(f"{w['module']:18s} C {w['components']:5d} rest {w['rest']:5d}  mass {w['mass_all']:.3f} rest {w['mass_rest']:.3f}  "
              f"rest_net {w['rest_net']:.3f}  delta {w['delta']:.3f}  out: rest {w['output_rest']:.3f} own_err {w['output_own_error']:.3f} "
              f"union_err {w['output_union_error']:.3f} all_err {w['output_all_error']:.3f}", flush=True)
    print('VPD_UNION_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
