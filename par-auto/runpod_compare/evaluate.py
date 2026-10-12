"""Common held-out KLKeep curves using each method's native sensor.

Scores and rankings are computed on the intact target once. Masks keep the
globally highest-scoring components at each position across all 24 matrices.
As in upstream CI-masked evaluation, VPD's training-only residual/delta is
excluded: it has no learned importance score. All-kept reconstruction error
is retained in the primary curve rather than subtracted.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from common import ROOT, MODULES, credentials, target, tokens, save_json
from train_pair import VPD, capture, core, method


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('method', choices=['vpd', 'parauto'])
    parser.add_argument('--sequences', type=int, default=128)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args(); credentials(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    root = ROOT/'output'/('pilot' if args.pilot else 'train')/args.method
    saved = torch.load(root/'final.pth', map_location='cpu', weights_only=False)
    if not args.pilot and saved['completed'] != 30518: raise ValueError('Expected completed 1B-position run')
    model = target()
    if args.method == 'vpd':
        # Evaluation needs the component/CI model, not optimizer/PGD tensors.
        trainer = VPD(model, evaluation=True)
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        from param_decomp.models.components import make_mask_infos
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    del saved
    data, manifest = tokens('val')
    fractions = [0., .0001, .0003, .001, .003, .01, .03, .1, .2, .4, .6, .8, 1.]
    sums = [dict(kl=0., ce=0., kept=0., n=0) for _ in fractions]
    baseline_ce = 0.; started = time.perf_counter()
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:min(first+2, args.sequences)], dtype=np.int64), device='cuda')
        ids, labels = rows[:, :-1], rows[:, 1:]
        if args.method == 'parauto':
            cache = capture(model, rows)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores = [method.scores_of(e, p).reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
            del cache
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if args.method == 'vpd':
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n] for n in MODULES]
                baseline = base.output
                del base, ci
            else: baseline = model(ids)[0]
            counts = [s.shape[-1] for s in scores]
            combined = torch.cat(scores, -1).float()
            # Stable sorting gives a deterministic tie-break for VPD's zeros.
            order = combined.argsort(dim=-1, descending=True, stable=True)
            del combined, scores
            logp = baseline.float().log_softmax(-1); prob = logp.exp()
            baseline_ce += float(F.cross_entropy(baseline.flatten(0, 1), labels.flatten(), reduction='sum'))
            del baseline
            for fraction, total in zip(fractions, sums):
                k = min(sum(counts), round(fraction*sum(counts)))
                keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
                masks = dict(zip(MODULES, keep.split(counts, -1)))
                if args.method == 'vpd':
                    infos = make_mask_infos(masks)
                    pred = trainer.model(ids, mask_infos=infos)
                else:
                    hooks = []
                    try:
                        for name, p in zip(MODULES, pieces):
                            def replace(layer, inputs, output, p=p, mask=masks[name]):
                                value = ((inputs[0] @ p.read)*mask) @ p.write.T
                                return value if layer.bias is None else value+layer.bias
                            hooks.append(model.get_submodule(name).register_forward_hook(replace))
                        pred = model(ids)[0]
                    finally:
                        for hook in hooks: hook.remove()
                # Chunk the vocabulary calculation by positions; retain no
                # training graph and accumulate KL in FP32 for both methods.
                kl = (prob*(logp-pred.float().log_softmax(-1))).sum()
                total['kl'] += float(kl)
                total['ce'] += float(F.cross_entropy(pred.flatten(0, 1), labels.flatten(), reduction='sum'))
                total['kept'] += k*ids.numel(); total['n'] += ids.numel()
                del pred, masks, keep
        print(json.dumps(dict(evaluated_sequences=min(first+2, args.sequences))), flush=True)
    curve = [dict(keep_fraction=f, kept_components=s['kept']/s['n'], kl=s['kl']/s['n'],
                  ce=s['ce']/s['n']) for f, s in zip(fractions, sums)]
    result = dict(method=args.method, curve=curve, baseline_ce=baseline_ce/(args.sequences*512),
                  positions=args.sequences*512, total_components=sum(counts),
                  delta_components_included=False, sensor='scalar CE' if args.method == 'parauto' else 'VPD CI',
                  seconds=time.perf_counter()-started, data_sha256=manifest['val']['sha256'])
    save_json(result, root/'klkeep.json')
    run_info = json.loads((root/'wandb.json').read_text())
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu', id=run_info['id'], resume='must')
    table = wandb.Table(columns=['kept_components', 'kl', 'ce'], data=[[r[k] for k in ('kept_components', 'kl', 'ce')] for r in curve])
    run.log({'eval/klkeep': wandb.plot.line(table, 'kept_components', 'kl', title='KLKeep')})
    run.summary['evaluation'] = result; run.finish()


if __name__ == '__main__': main()
