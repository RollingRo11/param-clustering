"""vpd_union_probe67.py for either method: what do the pieces that no donor token needs do? In aggregation67.py's
'all texts' rung (mask 1 on the union U of the needed sets of all 524,288 donor positions, the recipient's own
explanation elsewhere) the KL differs from every piece on only through R, the pieces never needed at any donor position.

Per matrix: R's share of the weight (sum of ||v_c||^2 ||u_c||^2, and the norm of R's summed matrix, both over ||W||_F)
and, on the recipients' intact inputs, R's share of the matrix output, next to the output error of the own explanation
and of the union-on mask. Then the KL (as aggregation67.py: KL(P_intact || Q_masked) per position, mean over a text's
512 positions, then over texts) for masks that switch R on or off around the union and the own explanation, overall,
per layer and per matrix type. Explanations as aggregation67.py: VPD its labels (needed: > 0.1, delta left out);
par-auto the top-k pieces by the Fisher K=8 score (needed: in the top k).
Usage: union_probe67.py --method parauto|vpd --checkpoint PATH --output JSON [--budgets 169,430] [--recipients 1024] [--donors 1024]"""
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
from train_pair import core, method

POSITIONS = 512
RECIPIENTS, DONORS = 0, 1024           # first val rows of each, as in aggregation67.py
NEEDED = 0.1
FISHER_K = 8


def bootstrap(values, generator, resamples=10000):
    """Mean of a per-text array and the 95% interval of that mean, resampling texts."""
    v = torch.as_tensor(values, dtype=torch.float64)
    means = v[torch.randint(len(v), (resamples, len(v)), generator=generator)].mean(-1)
    return dict(mean=float(v.mean()), lo=float(means.quantile(.025)), hi=float(means.quantile(.975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['parauto', 'vpd'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--budgets', default='169,430', help='par-auto explanation sizes, pieces per token')
    ap.add_argument('--recipients', type=int, default=1024); ap.add_argument('--donors', type=int, default=1024)
    ap.add_argument('--chunk', type=int, default=16, help='texts per forward')
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    started = time.perf_counter()
    vpd = args.method == 'vpd'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'))
    model = target()
    if vpd:
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        from param_decomp.models.components import make_mask_infos
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        counts = [trainer.model.module_to_c[n] for n in MODULES]
        # (read, write) per matrix so that a matrix's masked output is ((x @ read) * mask) @ write.T, and its weight [d_out, d_in].
        factors = [(trainer.model.components[n].V.detach(), trainer.model.components[n].U.detach().T) for n in MODULES]
        targets = [trainer.model.target_weight(n).detach() for n in MODULES]
    else:
        from fisher_attribution import capture_fisher
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
        counts = [p.read.shape[1] for p in pieces]
        factors = [(p.read, p.write) for p in pieces]
        targets = []
        for n in MODULES:
            m = model.get_submodule(n)
            targets.append((m.weight.T if type(m).__name__ == 'RadfordConv1D' else m.weight).detach())
    del saved
    total = sum(counts)
    for (read, write), w in zip(factors, targets): assert w.shape == (write.shape[0], read.shape[0]), (w.shape, read.shape, write.shape)
    budgets = [None] if vpd else [int(b) for b in args.budgets.split(',')]
    key_of = lambda b: 'vpd_labels' if b is None else f'top{b}'
    kinds = list(dict.fromkeys(n.split('.', 2)[2] for n in MODULES))
    matrix_of = torch.cat([torch.full((c,), i) for i, c in enumerate(counts)]).cuda()
    layer_of = torch.cat([torch.full((c,), int(n.split('.')[1])) for n, c in zip(MODULES, counts)]).cuda()
    kind_of = torch.cat([torch.full((c,), kinds.index(n.split('.', 2)[2])) for n, c in zip(MODULES, counts)]).cuda()
    data, manifest = tokens('val')
    assert len(data) >= DONORS + args.donors and args.recipients <= DONORS

    def rows_at(first, n):
        return torch.tensor(np.array(data[first:first+n], dtype=np.int64), device='cuda')

    def explain(rows, first):
        """Scores [n, 512, total] fp32 (VPD labels in [0, 1]; par-auto Fisher K=8) and each matrix's intact input [n*512, d_in]."""
        ids = rows[:, :-1]
        if vpd:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                out = torch.cat([ci[n] for n in MODULES], -1).float().clamp(0., 1.)
            return out, [base.cache[n].flatten(0, 1) for n in MODULES]
        cache = capture_fisher(model, rows, k=FISHER_K, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = torch.cat([method.scores_of(e, p).reshape(*ids.shape, -1) for e, p in zip(cache, pieces)], -1).float()
        return out, [e['h'] for e in cache]

    def own_of(e, b):
        return e if b is None else torch.zeros_like(e).scatter_(-1, e.topk(b, dim=-1, sorted=False).indices, 1.)

    def forward(ids, mask):
        """Logits with every matrix replaced by its masked pieces (VPD: masked components, delta left out)."""
        parts = mask.split(counts, -1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if vpd: return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, parts)))).float()
            hooks = []
            try:
                for name, p, m in zip(MODULES, pieces, parts):
                    hooks.append(model.get_submodule(name).register_forward_hook(
                        lambda _layer, inputs, _out, p=p, m=m: ((inputs[0] @ p.read) * m) @ p.write.T))
                return model(ids)[0].float()
            finally:
                for h in hooks: h.remove()

    # 1. Donors: U = needed at >= 1 donor position (VPD label > 0.1; par-auto in the top k); R = the rest.
    unions = {b: torch.zeros(total, dtype=torch.bool, device='cuda') for b in budgets}
    for first in range(DONORS, DONORS + args.donors, args.chunk):
        n = min(args.chunk, DONORS + args.donors - first)
        e, _ = explain(rows_at(first, n), first)
        e = e.flatten(0, 1)
        if vpd: unions[None] |= (e > NEEDED).any(0)
        else:
            top = e.topk(max(budgets), dim=-1, sorted=True).indices
            for b in budgets: unions[b][top[:, :b].flatten()] = True
            del top
        del e
    rests = {b: ~u for b, u in unions.items()}
    print(dict(donor_positions=args.donors * POSITIONS, total=total, union={key_of(b): int(u.sum()) for b, u in unions.items()},
               seconds=round(time.perf_counter() - started)), flush=True)

    # 2. Weights: R's share of each matrix.
    weights = {b: [] for b in budgets}
    for i, name in enumerate(MODULES):
        read, write = factors[i][0].double(), factors[i][1].double()                 # [d_in, C], [d_out, C]
        W = targets[i].double()
        size = read.pow(2).sum(0) * write.pow(2).sum(0)                              # ||r_c||^2 ||u_c||^2
        w2 = float(W.pow(2).sum())
        for b in budgets:
            r, u = rests[b][matrix_of == i], unions[b][matrix_of == i]
            weights[b].append(dict(module=name, components=counts[i], union=int(u.sum()), rest=int(r.sum()),
                                   mass_all=float(size.sum()) / w2, mass_rest=float(size[r].sum()) / w2,
                                   rest_net=float((write[:, r] @ read[:, r].T).norm()) / w2**.5,
                                   union_net_error=float((write[:, u] @ read[:, u].T - W).norm()) / w2**.5,
                                   delta=float((write @ read.T - W).norm()) / w2**.5,
                                   rest_largest_mass=float(size[r].max()) / w2 if r.any() else 0.))
        del read, write, W

    # 3. Recipients: per matrix output shares at the intact inputs, and the KL of each mask.
    def masks_of(b):
        union, rest = unions[b], rests[b]
        masks = {
            'own': lambda own: own,
            'all_on': lambda own: torch.ones_like(own),
            'union_on': lambda own: torch.where(union, 1., own),                 # aggregation67.py's 'all texts' rung
            'union_on_rest_zero': lambda own: torch.where(union, 1., torch.zeros_like(own)),
            'union_on_rest_half': lambda own: torch.where(union, 1., torch.full_like(own, .5)),
            'own_rest_on': lambda own: torch.where(rest, 1., own),
            'own_rest_zero': lambda own: torch.where(rest, 0., own),
        }
        for L in range(4):
            masks[f'union_on_rest_on_layer{L}'] = lambda own, L=L: torch.where(union | (rest & (layer_of == L)), 1., own)
            masks[f'own_union_on_layer{L}'] = lambda own, L=L: torch.where(union & (layer_of == L), 1., own)
        for k, kind in enumerate(kinds):
            masks[f'union_on_rest_on_{kind}'] = lambda own, k=k: torch.where(union | (rest & (kind_of == k)), 1., own)
        return masks
    masks = {b: masks_of(b) for b in budgets}
    kl = {b: {name: [] for name in masks[b]} for b in budgets}
    out_sq = {b: {key: torch.zeros(len(MODULES), dtype=torch.float64) for key in ('target', 'rest', 'own_error', 'union_error', 'all_error')}
              for b in budgets}
    for first in range(RECIPIENTS, RECIPIENTS + args.recipients, args.chunk):
        tick = time.perf_counter()
        n = min(args.chunk, RECIPIENTS + args.recipients - first)
        rows = rows_at(first, n); ids = rows[:, :-1]
        e, inputs = explain(rows, first)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16): logp = model(ids)[0].float().log_softmax(-1)
            p = logp.exp()
            for b in budgets:
                own = own_of(e, b)
                for i in range(len(MODULES)):
                    read, write = factors[i][0].float(), factors[i][1].float()
                    x = inputs[i].float()
                    acts = x @ read                                                      # [n*512, C]
                    y = x @ targets[i].float().T
                    g, r, u = own[..., matrix_of == i].flatten(0, 1), rests[b][matrix_of == i], unions[b][matrix_of == i]
                    for key, value in (('target', y), ('rest', acts[:, r] @ write[:, r].T), ('own_error', (acts * g) @ write.T - y),
                                       ('union_error', (acts * torch.where(u, 1., g)) @ write.T - y), ('all_error', acts @ write.T - y)):
                        out_sq[b][key][i] += float(value.double().pow(2).sum())
                    del x, acts, y, g
                for name, make in masks[b].items():
                    q = forward(ids, make(own)).log_softmax(-1)
                    kl[b][name].append((p * (logp - q)).sum(-1).mean(-1).cpu())
                    del q
                del own
        del e, inputs, logp, p
        print(dict(recipients_done=first + n - RECIPIENTS, chunk_seconds=round(time.perf_counter() - tick, 1),
                   **{key_of(b): dict(own=round(float(torch.cat(kl[b]['own']).mean()), 4), union_on=round(float(torch.cat(kl[b]['union_on']).mean()), 4),
                                      all_on=round(float(torch.cat(kl[b]['all_on']).mean()), 4)) for b in budgets}), flush=True)

    generator = torch.Generator().manual_seed(0)
    summary = {}
    for b in budgets:
        values = {name: torch.cat(v) for name, v in kl[b].items()}
        results = {name: dict(kl=bootstrap(v, generator), rise_over_own=bootstrap(v - values['own'], generator)) for name, v in values.items()}
        for i, w in enumerate(weights[b]):
            t = float(out_sq[b]['target'][i])
            w.update({f'output_{key}': (float(out_sq[b][key][i]) / t)**.5 for key in ('rest', 'own_error', 'union_error', 'all_error')})
        summary[key_of(b)] = dict(union=int(unions[b].sum()), rest=int(rests[b].sum()),
                                  per_layer={L: dict(union=int((unions[b] & (layer_of == L)).sum()), rest=int((rests[b] & (layer_of == L)).sum())) for L in range(4)},
                                  weights=weights[b], results=results)
    result = dict(method=args.method, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta,
                  recipients=[RECIPIENTS, RECIPIENTS + args.recipients], donors=[DONORS, DONORS + args.donors], positions_per_text=POSITIONS,
                  data_sha256=manifest['val']['sha256'], total=total, gpu=torch.cuda.get_device_name(), seconds=time.perf_counter() - started,
                  results=summary)
    save_json(result, args.output)
    for key, s in summary.items():
        print(key, dict(union=s['union'], rest=s['rest']), flush=True)
        for name, r in s['results'].items():
            print(f"  {name:32s} KL {r['kl']['mean']:.4f}  rise {r['rise_over_own']['mean']:+.4f}", flush=True)
        for w in s['weights']:
            print(f"  {w['module']:18s} C {w['components']:5d} rest {w['rest']:5d}  mass {w['mass_all']:.3f} rest {w['mass_rest']:.3f}  "
                  f"rest_net {w['rest_net']:.3f}  delta {w['delta']:.3f}  out: rest {w['output_rest']:.3f} own_err {w['output_own_error']:.3f} "
                  f"union_err {w['output_union_error']:.3f} all_err {w['output_all_error']:.3f}", flush=True)
    print('UNION_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
