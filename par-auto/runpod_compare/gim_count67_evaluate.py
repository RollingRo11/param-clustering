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

from pathlib import Path
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import VPD, capture, core, method


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('method', choices=['vpd', 'parauto'])
    parser.add_argument('--sequences', type=int, default=128)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, default=2000)
    parser.add_argument('--run-tag', default=None)
    parser.add_argument('--sensor', choices=['gradient','ig','gim','write','finite','fisher','fisher_gim'], default='gradient')
    parser.add_argument('--ig-k', type=int, default=5)
    parser.add_argument('--fisher-k', type=int, default=8)
    parser.add_argument('--start-sequence', type=int, default=256)
    parser.add_argument('--keep-counts', default='0,64,128,256,389,430,512,1024,2048,4096,8192,16384')
    args = parser.parse_args(); credentials(); torch.set_num_threads(4)
    if args.method == 'vpd' and args.sensor != 'gradient': raise ValueError('VPD uses its native CI sensor')
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if saved['completed'] != args.expected_step or saved['method'] != args.method or saved['target_sha256'] != TARGET_SHA:
        raise ValueError('Checkpoint identity mismatch')
    train_sha = saved['data_sha256']
    provenance = saved.get('provenance', {'kind': 'original checkpoint'})
    training_positions = saved.get('training_positions', args.expected_step*32768)
    checkpoint_sha = sha(args.checkpoint)
    model = target()
    if args.method == 'vpd':
        # Evaluation needs the component/CI model, not optimizer/PGD tensors.
        trainer = VPD(model, evaluation=True)
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        handle_deprecated_state_dict_keys_(saved['factors'])
        # A published checkpoint also embeds target weights. Check these before
        # loading so both curves really use the identical frozen target.
        for name, value in model.state_dict().items():
            torch.testing.assert_close(saved['factors']['target_model.'+name], value.cpu(), rtol=0, atol=0)
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        from param_decomp.models.components import make_mask_infos
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    # Rank-k pieces (training_config.piece_rank > 1): rank and mask whole pieces; columns follow their piece.
    piece_rank = provenance.get('training_config', {}).get('piece_rank', 1) if args.method == 'parauto' else 1
    if piece_rank > 1 and not args.sensor.startswith(('gim', 'fisher')): raise ValueError('rank-k pieces need a vjp sensor')
    # Owned components (factors carry ownership_logits): each piece belongs to its argmax component; empty components drop out.
    owners = None
    if args.method == 'parauto' and 'ownership_logits' in saved['factors'][0]:
        if not args.sensor.startswith(('gim', 'fisher')): raise ValueError('owned components need a gim or fisher sensor')
        owners = []
        for factor in saved['factors']:
            used, compact = torch.unique(factor['ownership_logits'].argmax(-1), return_inverse=True)
            owners.append((compact.cuda(), len(used)))
    del saved
    data, manifest = tokens('val')
    total_capacity = 38912 if args.method == 'vpd' else sum(p.count for p in pieces)
    if owners is not None or piece_rank != 1: raise ValueError('This experiment compares primitive rank-one pieces')
    budgets = sorted(set([int(k) for k in args.keep_counts.split(',')] + [total_capacity]))
    if any(k < 0 or k > total_capacity for k in budgets): raise ValueError('Invalid retained count')
    if not 0 <= args.start_sequence < len(data) or args.start_sequence+args.sequences > len(data): raise ValueError('Invalid evaluation slice')
    fractions = [k/total_capacity for k in budgets]
    sums = [dict(kl=0., ce=0., kept=0., n=0, sequence_kl=[]) for _ in fractions]
    baseline_ce = 0.; started = time.perf_counter(); integration_audits = []; scoring_seconds = 0.
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[args.start_sequence+first:args.start_sequence+min(first+2, args.sequences)], dtype=np.int64), device='cuda')
        ids, labels = rows[:, :-1], rows[:, 1:]
        if args.method == 'parauto':
            torch.cuda.synchronize(); score_start=time.perf_counter()
            if args.sensor == 'ig':
                from integrated_attribution import integrated_scores
                scores, audit = integrated_scores(model, rows, pieces, k=args.ig_k)
                integration_audits.append(audit)
            else:
                if args.sensor == 'gim':
                    from gim_attribution import capture_gim
                    cache = capture_gim(model, rows)
                elif args.sensor == 'write':
                    from write_attribution import capture_inputs
                    cache = capture_inputs(model, rows)
                elif args.sensor == 'finite':
                    from finite_ablation import capture_finite
                    cache = capture_finite(model, rows)
                elif args.sensor.startswith('fisher'):
                    from fisher_attribution import capture_fisher
                    cache = capture_fisher(model, rows, k=args.fisher_k, seed=500_000+args.start_sequence+first, gim=args.sensor == 'fisher_gim')
                else:
                    cache = capture(model, rows)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    if owners is not None:
                        scores = [method.structured.hard_component_scores(e, p, o, m).reshape(*ids.shape, -1) for e, p, (o, m) in zip(cache, pieces, owners)]
                    else:
                        scores = [method.scores_of(e, p, piece_rank).reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
                del cache
            torch.cuda.synchronize(); scoring_seconds+=time.perf_counter()-score_start
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
                parts = keep.split(counts, -1)
                if owners is not None: parts = [m.index_select(-1, o) for m, (o, _) in zip(parts, owners)]
                elif piece_rank > 1: parts = [m.repeat_interleave(piece_rank, -1) for m in parts]
                masks = dict(zip(MODULES, parts))
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
                per_sequence = (prob*(logp-pred.float().log_softmax(-1))).sum(-1).mean(-1)
                kl = per_sequence.sum()*ids.shape[1]
                total['sequence_kl'].extend(per_sequence.tolist())
                total['kl'] += float(kl)
                total['ce'] += float(F.cross_entropy(pred.flatten(0, 1), labels.flatten(), reduction='sum'))
                total['kept'] += k*ids.numel(); total['n'] += ids.numel()
                del pred, masks, keep
        print(json.dumps(dict(evaluated_sequences=min(first+2, args.sequences))), flush=True)
    curve = [dict(keep_fraction=f, kept_components=s['kept']/s['n'], kl=s['kl']/s['n'],
                  ce=s['ce']/s['n'], sequence_kl=s['sequence_kl']) for f, s in zip(fractions, sums)]
    result = dict(method=args.method, curve=curve, baseline_ce=baseline_ce/(args.sequences*512),
                  positions=args.sequences*512, total_components=sum(counts),
                  delta_components_included=False, sensor=(f'scalar CE integrated gradients k={args.ig_k}' if args.sensor == 'ig' else 'scalar CE GIM' if args.sensor == 'gim' else 'write size |r.h| ||u||' if args.sensor == 'write' else 'per-piece finite local ablation' if args.sensor == 'finite' else f'Fisher k={args.fisher_k} sampled labels' + (' + GIM rules' if args.sensor == 'fisher_gim' else '') if args.sensor.startswith('fisher') else 'scalar CE') if args.method == 'parauto' else 'VPD CI',
                  seconds=time.perf_counter()-started, data_sha256=manifest['val']['sha256'])
    if integration_audits:
        estimate=sum(a['signed_attribution_sum'] for a in integration_audits)/(args.sequences*512)
        true_delta=curve[-1]['ce']-curve[0]['ce']
        result['integration']=dict(**{k:v for k,v in integration_audits[0].items() if k!='signed_attribution_sum'},
                 integrated_ce_change_per_position=estimate,actual_ce_change_per_position=true_delta,
                 completeness_relative_error=abs(estimate-true_delta)/max(abs(true_delta),1e-8),
                 endpoint_note='Actual endpoints use factorized all-kept / all-removed forwards; path uses target matrices')
    result['scoring_seconds']=scoring_seconds
    result['start_sequence']=args.start_sequence
    result['keep_counts']=budgets
    result['ranking_seed']='500000+absolute_first_sequence'
    save_json(result, root/'klkeep.json')
    result.update(training_steps=args.expected_step, training_positions=training_positions,
                  checkpoint_provenance=provenance,
                  training_data_sha256=train_sha, checkpoint_sha256=checkpoint_sha, target_sha256=TARGET_SHA,
                  precision='bf16 autocast; fp32 KL reductions', gpu=torch.cuda.get_device_name())
    save_json(result, root/'klkeep.json')
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu',
                     name=args.run_tag or f'paired-1b-{args.method}-klkeep-{args.expected_step}',
                     group='gim-count67-20260914', job_type='evaluation', dir=str(root),
                     config={k: v for k, v in result.items() if k != 'curve'})
    save_json(dict(id=run.id, url=run.url), root/'wandb.json')
    table = wandb.Table(columns=['kept_components', 'kl', 'ce'], data=[[r[k] for k in ('kept_components', 'kl', 'ce')] for r in curve])
    run.log({'eval/klkeep': wandb.plot.line(table, 'kept_components', 'kl', title='KLKeep')})
    run.summary['evaluation'] = result; run.finish()


if __name__ == '__main__': main()
