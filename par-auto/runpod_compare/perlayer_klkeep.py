"""Per-layer KLKeep: mask only one layer's six matrices at each keep fraction, everything else intact.

Ranking is the method's native one on the intact target (VPD: CI; parauto: --sensor gim|fisher),
global across the masked layer's matrices at each position. Also records the usual all-layer curve.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto'); ap.add_argument('--sensor', choices=['gim', 'fisher'], default='gim')
    ap.add_argument('--sequences', type=int, default=128); ap.add_argument('--fisher-k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'), provenance=saved.get('provenance'))
    model = target()
    if args.method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        from param_decomp.models.components import make_mask_infos
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    del saved
    data, manifest = tokens('val')
    fractions = [.0001, .0003, .001, .003, .01, .03, .1, .2, .4, .6, .8, 1.]
    layer_of = [int(n.split('.')[1]) for n in MODULES]
    conditions = {'all': lambda m: True, **{f'layer{l}': (lambda m, l=l: layer_of[m] == l) for l in range(4)}}
    sums = {(c, f): 0. for c in conditions for f in fractions}; n_tok = 0; started = time.perf_counter()

    def masked_forward(ids, masks):
        if args.method == 'vpd':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, masks))))
        hooks = []
        try:
            for name, p, mask in zip(MODULES, pieces, masks):
                def replace(layer, inputs, output, p=p, mask=mask):
                    return ((inputs[0] @ p.read)*mask) @ p.write.T
                hooks.append(model.get_submodule(name).register_forward_hook(replace))
            with torch.autocast('cuda', dtype=torch.bfloat16): return model(ids)[0]
        finally:
            for h in hooks: h.remove()

    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        if args.method == 'parauto':
            if args.sensor == 'gim':
                from gim_attribution import capture_gim
                cache = capture_gim(model, rows)
            else:
                from fisher_attribution import capture_fisher
                cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if args.method == 'vpd':
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n].float() for n in MODULES]; del base, ci
            else:
                scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]; del cache
            baseline = model(ids)[0]
        with torch.no_grad():
            counts = [s.shape[-1] for s in scores]
            logp = baseline.float().log_softmax(-1); prob = logp.exp(); del baseline
            for cname, active in conditions.items():
                idx = [m for m in range(len(MODULES)) if active(m)]
                combined = torch.cat([scores[m] for m in idx], -1)                       # rank within the masked set
                order = combined.argsort(dim=-1, descending=True, stable=True)
                for f in fractions:
                    k = min(combined.shape[-1], round(f*combined.shape[-1]))
                    keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
                    parts = dict(zip(idx, keep.split([counts[m] for m in idx], -1)))
                    masks = [parts[m] if m in parts else torch.ones(*ids.shape, counts[m], device='cuda') for m in range(len(MODULES))]
                    pred = masked_forward(ids, masks)
                    sums[(cname, f)] += float((prob*(logp-pred.float().log_softmax(-1))).sum()); del pred, masks, keep
            n_tok += ids.numel()
        print(json.dumps(dict(sequences=first+2, seconds=time.perf_counter()-started)), flush=True)
    curves = {c: [dict(keep_fraction=f, kl=sums[(c, f)]/n_tok) for f in fractions] for c in conditions}
    save_json(dict(method=args.method, sensor=('VPD CI' if args.method == 'vpd' else args.sensor), checkpoint=str(args.checkpoint),
                   checkpoint_sha256=sha(args.checkpoint), **meta, positions=n_tok, gpu=torch.cuda.get_device_name(),
                   data_sha256=manifest['val']['sha256'], components_per_layer=[sum(counts[m] for m in range(len(MODULES)) if layer_of[m] == l) for l in range(4)],
                   curves=curves, seconds=time.perf_counter()-started), args.output)
    for c, rows_ in curves.items(): print(c, ' '.join(f"{r['keep_fraction']:g}:{r['kl']:.3g}" for r in rows_), flush=True)
    print('PERLAYER_DONE', flush=True)


if __name__ == '__main__': main()
