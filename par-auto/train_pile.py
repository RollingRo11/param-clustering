"""Cache, train/resume, and verify par-auto on the existing 4L Pile target."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import parauto_core as core
import parauto_method as method
import parauto_pile as pile
import parauto_structured as structured
import parauto_tracking as tracking
from parauto_target import atomic_json, atomic_save, fingerprint


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('cache', 'train', 'evaluate'))
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--tokens', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--probes', type=int, default=8)
    parser.add_argument('--train-stop', type=int, default=100000000)
    parser.add_argument('--eval-blocks', type=int, default=4)
    parser.add_argument('--components', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--max-steps', type=int, default=None, help='Stop cleanly after this many updates; preserve full LR/p schedule')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    tracking.start_cli(args)
    tracking.configure(matrix_names=pile.MODULES)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model = pile.load_model(args.target)
    if args.phase == 'cache':
        manifest = pile.make_cache(model, args.target, args.tokens, args.cache, blocks=args.blocks,
                                   context=args.context, probes=args.probes, train_stop=args.train_stop, seed=args.seed)
        print(json.dumps({'CACHE_COMPLETE': manifest}), flush=True)
        return
    if args.phase == 'train': entries, manifest = pile.load_cache(args.cache, args.target, 'cuda')
    else: entries, manifest = [], pile.cache_manifest(args.cache, args.target)
    weights = [model.get_submodule(name).weight.detach() for name in pile.MODULES]
    cache_sha = fingerprint(args.cache/'manifest.json')
    if args.components <= max(min(w.shape) for w in weights): raise ValueError('Use more components than each matrix rank bound (768)')
    cfg = method.TrainConfig(steps=args.steps, batch=args.batch, seed=args.seed, lr=args.lr, lr_min=args.lr_min,
                             precision='bf16', projection_precision='fp32', log_interval=100)
    tracking.configure({'training_config': asdict(cfg)})
    checkpoint_path = args.output/'checkpoint.pt'
    if args.phase == 'train':
        resume = None
        if args.resume:
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if saved['cache_sha256'] != cache_sha or saved['target_sha256'] != manifest['target_sha256']:
                raise ValueError('Resume target/cache mismatch')
            resume = saved['training']
            if any(p['read'].shape[1] != args.components for p in resume['pieces']):
                raise ValueError('Resume component count mismatch')
            initial = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in resume['pieces']]
        else:
            if checkpoint_path.exists(): raise FileExistsError('Checkpoint exists; pass --resume explicitly')
            initial = []
            for name, entry, weight in zip(pile.MODULES, entries, weights):
                print(f'INITIALIZE {name}', flush=True)
                item, info = method.prepare_initial(weight, args.components, cfg,
                                                     log=lambda row: print(json.dumps({'initialization': name, **row}), flush=True))
                initial.append(item)
                tracking.log_initialization(name, info)
        next_step = resume['next_step'] if resume else 0
        def checkpoint(state):
            nonlocal next_step
            next_step = state['next_step']
            atomic_save(cpu_tree(dict(format='parauto_pile_checkpoint_v1', target_sha256=manifest['target_sha256'],
                                      cache_sha256=cache_sha, modules=list(pile.MODULES),
                                      method_source_sha256=fingerprint(method.__file__), training=state)), checkpoint_path)
            print(json.dumps({'checkpoint_step': next_step}), flush=True)
            tracking.log_metrics({'checkpoint_step': next_step}, prefix='checkpoint')
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        pieces, histories = method.fit(entries, weights, initial, cfg, resume=resume, checkpoint=checkpoint,
                                       checkpoint_interval=args.checkpoint_interval, stop_after=args.max_steps,
                                       log=lambda row: print(json.dumps({'module': pile.MODULES[row['matrix_index']], **row}), flush=True))
        torch.cuda.synchronize(); elapsed = time.perf_counter()-started
        errors = {n: core.reconstruction_error(w, p) for n, w, p in zip(pile.MODULES, weights, pieces)}
        if max(errors.values()) > 1e-5: raise RuntimeError(f'Matrix sum mismatch: {errors}')
        atomic_save({'pieces': [{'read': p.read.cpu(), 'write': p.write.cpu()} for p in pieces]}, args.output/'pieces.pt')
        result = dict(format='parauto_pile_fit_v1', optimizer_version='projected_adam_no_rejection_v1', target_sha256=manifest['target_sha256'], cache_sha256=cache_sha,
                      modules=list(pile.MODULES), components_per_matrix=args.components, config=asdict(cfg),
                      next_step=next_step, complete=next_step == cfg.steps, invocation_seconds=elapsed,
                      cache_bytes=structured.cache_bytes(dict(zip(pile.MODULES, entries))),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(),
                      matrix_sum_errors=errors, precision='BF16 cache/score intermediates; FP32 factors/Adam/projection and its backward',
                      sensor=manifest['sensor'], model_source_sha256=manifest['model_source_sha256'])
        atomic_json(result, args.output/'result.json')
        print(json.dumps({'PILE_TRAIN_COMPLETE': result}), flush=True)
    else:
        stored = torch.load(args.output/'pieces.pt', weights_only=True, map_location='cuda')['pieces']
        pieces = [core.FreePieces(p['read'], p['write']) for p in stored]
        result = json.loads((args.output/'result.json').read_text())
        if result['target_sha256'] != manifest['target_sha256'] or result['cache_sha256'] != cache_sha: raise ValueError('Evaluation checkpoint mismatch')
        if fingerprint(args.tokens) != manifest['token_file_sha256']: raise ValueError('Evaluation corpus changed')
        ids, offsets = pile.sample_blocks(args.tokens, args.eval_blocks, manifest['context'], manifest['train_range'][1],
                                          args.tokens.stat().st_size//2, args.seed+920091)
        metrics = pile.evaluate(model, pieces, ids, manifest['probes'])
        all_kept = metrics['budgets'][str(sum(p.count for p in pieces))]['kl']
        if all_kept > 1e-6: raise RuntimeError(f'All-kept LM KL too large: {all_kept}')
        metrics.update(offsets=offsets, heldout_range=[manifest['train_range'][1], args.tokens.stat().st_size//2],
                       target_sha256=manifest['target_sha256'], checkpoint_sha256=fingerprint(args.output/'pieces.pt'))
        atomic_json(metrics, args.output/'evaluation.json')
        print(json.dumps({'PILE_EVALUATION_COMPLETE': metrics}), flush=True)


if __name__ == '__main__': tracking.run_main(main)
