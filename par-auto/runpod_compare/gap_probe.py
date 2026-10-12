"""Where does ablatability live: in the pieces, or in the masked network?

For a checkpoint (parauto or vpd), at keep fractions 1% and 3%, native ranking (Fisher K=8 / CI):
  all         mask every matrix (KLKeep as usual)
  layer{l}    mask only layer l's six matrices; every other matrix fully kept
  attn, mlp   mask only attention / only MLP matrices
  sum_layers  sum of the four single-layer KLs (additivity reference)
If single-layer masking hurts MORE than masking everything, the layers compensate for each other
under masks, i.e. the trained masked network is faithful even though its layers are not.

Also, exact (linear) relative output change at h.0.attn.{q,k,v}_proj, whose inputs are identical
in the masked and intact models: ||sum_removed (r.h) u|| / ||W h||, averaged over tokens.
For vpd, an extra condition ranks VPD's own pieces by our Fisher K=8 score instead of its CI.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method
from fisher_attribution import capture_fisher


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto'); ap.add_argument('--sequences', type=int, default=128)
    ap.add_argument('--expected-step', type=int, default=5000)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['completed'] == args.expected_step and saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    model = target()
    if args.method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        from param_decomp.models.components import make_mask_infos
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    del saved
    data, manifest = tokens('val')
    fractions = [.01, .03]
    layer_of = [int(n.split('.')[1]) for n in MODULES]
    is_attn = ['attn' in n for n in MODULES]
    conditions = {'all': lambda m: True, 'attn': lambda m: is_attn[m], 'mlp': lambda m: not is_attn[m],
                  **{f'layer{l}': (lambda m, l=l: layer_of[m] == l) for l in range(4)}}
    rankings = ['native'] + (['fisher'] if args.method == 'vpd' else [])
    sums = {(rk, f, c): 0. for rk in rankings for f in fractions for c in conditions}
    rel = {f: dict(zip(('q', 'k', 'v'), [0., 0., 0.])) for f in fractions}
    n_tok = 0; started = time.perf_counter()

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
        cache = capture_fisher(model, rows, k=8, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            fisher_scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
            if args.method == 'vpd':
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                native = [ci.lower_leaky[n].float() for n in MODULES]; del base, ci
            else:
                native = fisher_scores
            baseline = model(ids)[0]
        h0 = [cache[MODULES.index(f'h.0.attn.{s}_proj')]['h'].float() for s in 'qkv']
        del cache
        with torch.no_grad():
            counts = [s.shape[-1] for s in native]
            logp = baseline.float().log_softmax(-1); prob = logp.exp(); del baseline
            for rk, scores in (('native', native), ('fisher', fisher_scores)):
                if rk not in rankings: continue
                order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)
                for f in fractions:
                    k = round(f*sum(counts)); keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
                    masks = list(keep.split(counts, -1))
                    for cname, active in conditions.items():
                        if rk == 'fisher' and cname != 'all': continue
                        cond_masks = [m if active(i) else torch.ones_like(m) for i, m in enumerate(masks)]
                        pred = masked_forward(ids, cond_masks)
                        sums[(rk, f, cname)] += float((prob*(logp-pred.float().log_softmax(-1))).sum()); del pred
                    if rk == 'native':
                        for s, h in zip('qkv', h0):
                            m = MODULES.index(f'h.0.attn.{s}_proj'); p = pieces[m]
                            removed = ((h @ p.read)*(1-masks[m].reshape(-1, counts[m]))) @ p.write.T          # exact linear change
                            full = h @ model.get_submodule(MODULES[m]).weight.float().T
                            rel[f][s] += float((removed.norm(dim=-1)/full.norm(dim=-1).clamp_min(1e-12)).sum())
                    del keep, masks
            n_tok += ids.numel()
        print(json.dumps(dict(sequences=first+2, seconds=time.perf_counter()-started)), flush=True)
    result = {}
    for rk in rankings:
        for f in fractions:
            row = {c: sums[(rk, f, c)]/n_tok for c in conditions if (rk, f, c) in sums and not (rk == 'fisher' and c != 'all')}
            if rk == 'native': row['sum_layers'] = sum(row[f'layer{l}'] for l in range(4))
            result[f'{rk}_keep{f}'] = row
            print(json.dumps({f'{rk}_keep{f}': {k: round(v, 4) for k, v in row.items()}}), flush=True)
    result['layer0_relative_output_change'] = {str(f): {s: v/n_tok for s, v in rel[f].items()} for f in fractions}
    print(json.dumps(dict(layer0_relative_output_change=result['layer0_relative_output_change'])), flush=True)
    save_json(dict(method=args.method, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), positions=n_tok,
                   gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'], seconds=time.perf_counter()-started, **result), args.output)
    print('GAP_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
