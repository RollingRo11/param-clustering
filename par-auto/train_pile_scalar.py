"""Stream fresh scalar-CE batches through the joint 24-matrix decomposition.

The 1k-step profile and long run execute the identical cache/update/checkpoint
path. Only the scheduled update count differs. No full-corpus activation cache
is retained; each batch is used once before its activations are discarded.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import parauto_core as core
import parauto_method as method
import parauto_pile as pile
import parauto_pile_scalar as scalar
import parauto_tracking as tracking
from parauto_target import atomic_json, atomic_save, fingerprint
from train_pile import cpu_tree


def component_counts():
    counts = dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024,
                  c_fc=3072, down_proj=3584)
    return [counts[n.split('.')[-1]] for n in pile.MODULES]


def planned_steps(seconds_per_step, hours=8, reserve_seconds=900, slowdown=1.1):
    if seconds_per_step <= 0 or hours*3600 <= reserve_seconds or slowdown < 1:
        raise ValueError('Invalid profile or time budget')
    steps = int((hours*3600-reserve_seconds)/(seconds_per_step*slowdown))
    if steps < 100: raise ValueError('Profile leaves fewer than 100 updates')
    return steps//100*100


class TokenStream:
    """Epoch-shuffled, nonoverlapping 513-token blocks, reproducible by step."""
    def __init__(self, path, batch, context=512, train_stop=100000000, seed=0):
        self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
        if batch < context or batch % context:
            raise ValueError('Batch must be a positive multiple of context')
        if not context+1 <= train_stop <= len(self.tokens):
            raise ValueError('Training range outside corpus')
        self.batch, self.context, self.seed = batch, context, seed
        self.blocks_per_batch = batch//context
        self.available = train_stop//(context+1)
        self.epoch, self.order = None, None

    def blocks(self, step):
        if step < 0: raise ValueError('Negative stream step')
        indices = []
        for i in range(step*self.blocks_per_batch, (step+1)*self.blocks_per_batch):
            epoch, local = divmod(i, self.available)
            if epoch != self.epoch:
                self.order = np.random.default_rng(self.seed+epoch).permutation(self.available)
                self.epoch = epoch
            indices.append(int(self.order[local])*(self.context+1))
        blocks = torch.stack([torch.from_numpy(np.array(self.tokens[i:i+self.context+1], dtype=np.int64))
                              for i in indices])
        return blocks, indices


class ScalarBatches:
    def __init__(self, model, stream, microbatch=4):
        if microbatch < 1: raise ValueError('Invalid cache microbatch')
        self.model, self.stream, self.microbatch = model, stream, microbatch
        self.seconds, self.ce, self.offsets = 0., 0., []

    def __call__(self, step):
        torch.cuda.synchronize()
        start = time.perf_counter()
        blocks, self.offsets = self.stream.blocks(step)
        joined, total_ce = None, 0.
        for start_block in range(0, len(blocks), self.microbatch):
            block = blocks[start_block:start_block+self.microbatch].cuda()
            entries, ce = scalar.capture(self.model, block[:, :-1], block[:, 1:], torch.bfloat16)
            if joined is None:
                joined = [dict(e, h=torch.empty(self.stream.batch, e['h'].shape[-1], dtype=torch.bfloat16, device='cuda'),
                               events={'g': torch.empty(self.stream.batch, e['events']['g'].shape[-1], dtype=torch.bfloat16, device='cuda')},
                               net_rms=torch.zeros(self.stream.batch, device='cuda')) for e in entries]
            lo = start_block*self.stream.context
            hi = lo+block.shape[0]*self.stream.context
            for dst, src in zip(joined, entries, strict=True):
                dst['h'][lo:hi].copy_(src['h'])
                dst['events']['g'][lo:hi].copy_(src['events']['g'])
            total_ce += ce*len(block)
            del entries, block
        torch.cuda.synchronize()
        self.seconds += time.perf_counter()-start
        self.ce = total_ce/len(blocks)
        return joined


def main():
    entered = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    schedule = parser.add_mutually_exclusive_group(required=True)
    schedule.add_argument('--steps', type=int)
    schedule.add_argument('--profile-result', type=Path)
    parser.add_argument('--batch', type=int, default=32768)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--cache-microbatch', type=int, default=4)
    parser.add_argument('--train-stop', type=int, default=100000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--checkpoint-interval', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    tracking.start_cli(args)
    deadline = entered+args.hours*3600
    if os.environ.get('SLURM_JOB_END_TIME'):
        deadline = min(deadline, time.perf_counter()+int(os.environ['SLURM_JOB_END_TIME'])-time.time())
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This run requires native BF16 CUDA support')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    # Validate the actual corpus before model loading or initialization.
    stream = TokenStream(args.tokens, args.batch, args.context, args.train_stop, args.seed)
    token_sha = fingerprint(args.tokens)
    profile = None
    if args.profile_result:
        profile = json.loads(args.profile_result.read_text())
        expected = (args.batch, args.context, args.cache_microbatch, args.train_stop)
        observed = tuple(profile['stream'][k] for k in ('batch', 'context', 'cache_microbatch', 'train_stop'))
        if expected != observed or not profile['complete'] or profile['token_sha256'] != token_sha:
            raise ValueError('Profile/data configuration mismatch')
        if profile['gpu'] != torch.cuda.get_device_name(): raise ValueError('Profile GPU mismatch')
        if profile['source_sha256'] != fingerprint(__file__): raise ValueError('Profile runner source changed')
        if profile['method_source_sha256'] != fingerprint(method.__file__): raise ValueError('Profile method source changed')
        args.steps = planned_steps(profile['seconds_per_step_with_cache_and_checkpoints'], args.hours)
    if args.steps < 1 or args.checkpoint_interval < 1:
        raise ValueError('Positive step counts required')
    cfg = method.TrainConfig(steps=args.steps, batch=args.batch, lr=args.lr, lr_min=args.lr_min,
                             seed=args.seed, precision='bf16', projection_precision='fp32', log_interval=100)
    stream_config = dict(batch=args.batch, context=args.context, cache_microbatch=args.cache_microbatch,
                         train_stop=args.train_stop, seed=args.seed)
    tracking.configure({'training_config': asdict(cfg), 'stream': stream_config,
                        'component_counts': dict(zip(pile.MODULES, component_counts())),
                        'precision': 'BF16 cache and score autocast; FP32 model, parameters, Adam, projection/backward'},
                       matrix_names=pile.MODULES)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    path = output/'checkpoint.pt'
    if path.exists() and not args.resume: raise FileExistsError('Use --resume for an existing checkpoint')
    model = pile.load_model(args.target)
    weights = [model.get_submodule(n).weight for n in pile.MODULES]
    resume = None
    if args.resume:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['target_sha256'] != pile.TARGET_SHA or saved['token_sha256'] != token_sha or saved['stream'] != stream_config:
            raise ValueError('Resume target/data/stream mismatch')
        resume = saved['training']
        initial = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in resume['pieces']]
    else:
        initial = [core.initialize_pieces(w, c, args.seed) for w, c in zip(weights, component_counts(), strict=True)]
    capture = ScalarBatches(model, stream, args.cache_microbatch)
    # Keep only a small fixed, dispersed normalization set from the initial batch.
    normalization_batch = capture(0)
    ix = torch.randperm(args.batch, device='cuda', generator=torch.Generator(device='cuda').manual_seed(args.seed+202))[:cfg.normalization_events]
    references = [method.slice_entry(e, ix) for e in normalization_batch]
    pending = [normalization_batch] if resume is None else []
    del normalization_batch
    def provide(step):
        return pending.pop() if step == 0 and pending else capture(step)
    next_step = resume['next_step'] if resume else 0
    start_step = next_step
    stop = False
    def signal_stop(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGUSR1, signal_stop)
    signal.signal(signal.SIGTERM, signal_stop)
    samples, checkpoint_seconds = [], 0.
    baseline_cache_seconds = capture.seconds
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = last_tick = time.perf_counter()

    def finished_step(completed):
        nonlocal next_step, last_tick
        torch.cuda.synchronize()
        now = time.perf_counter()
        samples.append(now-last_tick)
        last_tick, next_step = now, completed
        if completed % 100 == 0:
            tracking.log_metrics({'positions_processed': completed*args.batch, 'cache_ce': capture.ce,
                                  'elapsed_seconds': now-started,
                                  'positions_per_second': (completed-start_step)*args.batch/(now-started)},
                                 prefix='performance', step=completed-1)
        return stop or now >= deadline-300

    def checkpoint(state):
        nonlocal checkpoint_seconds
        start = time.perf_counter()
        payload = dict(format='parauto_pile_scalar_stream_v1', target_sha256=pile.TARGET_SHA,
                       token_sha256=token_sha, stream=stream_config, training=state,
                       modules=list(pile.MODULES), method_source_sha256=fingerprint(method.__file__))
        atomic_save(cpu_tree(payload), path)
        if state['next_step'] in (1000, 5000) or state['next_step'] % 10000 == 0:
            atomic_save(cpu_tree(dict(pieces=state['pieces'], step=state['next_step'],
                                      target_sha256=pile.TARGET_SHA, config=asdict(cfg))),
                        output/f'model_{state["next_step"]}.pth')
        checkpoint_seconds += time.perf_counter()-start
        print(json.dumps({'checkpoint_step': state['next_step']}), flush=True)

    pieces, histories = method.fit(references, weights, initial, cfg, resume=resume,
                                   batch_provider=provide, step_callback=finished_step, checkpoint=checkpoint,
                                   checkpoint_interval=args.checkpoint_interval,
                                   log=lambda row: print(json.dumps(row), flush=True) if row['matrix_index'] == 0 else None)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    executed = next_step-start_step
    cache_elapsed = capture.seconds-baseline_cache_seconds
    # First 20 updates warm up allocator/libraries. This clock includes prior
    # checkpoint writes; add the final write via the whole-invocation average.
    steady = float(np.mean(samples[20:])) if len(samples) > 20 else elapsed/executed
    conservative = max(steady, elapsed/executed)
    errors = {n: core.reconstruction_error(w, p) for n, w, p in zip(pile.MODULES, weights, pieces, strict=True)}
    if max(errors.values()) > 1e-5: raise RuntimeError('Final exact-sum validation failed')
    atomic_save(cpu_tree({'pieces': [{'read': p.read, 'write': p.write} for p in pieces],
                          'step': next_step, 'target_sha256': pile.TARGET_SHA, 'config': asdict(cfg)}), output/'pieces.pt')
    result = dict(format='parauto_pile_scalar_stream_v1', complete=next_step == cfg.steps,
                  next_step=next_step, positions_processed=next_step*args.batch, config=asdict(cfg), stream=stream_config,
                  gpu=torch.cuda.get_device_name(), target_sha256=pile.TARGET_SHA, token_sha256=token_sha,
                  source_sha256=fingerprint(__file__), method_source_sha256=fingerprint(method.__file__),
                  component_counts=dict(zip(pile.MODULES, component_counts())), total_components=sum(component_counts()),
                  invocation_seconds=elapsed, setup_seconds=started-entered, cache_seconds=cache_elapsed,
                  checkpoint_seconds=checkpoint_seconds, fit_seconds=elapsed-cache_elapsed-checkpoint_seconds,
                  seconds_per_step_with_cache_and_checkpoints=conservative,
                  steady_seconds_per_step=steady, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  projected_8h_steps=planned_steps(conservative),
                  projection_errors=errors, final_total_loss=sum(h[-1]['loss'] for h in histories),
                  precision='BF16 cache/score autocast; FP32 frozen-model forward/backward, factors, Adam and projection',
                  sensor='context-summed ground-truth next-token CE derivative; no probes',
                  corpus_positions=len(stream.tokens), heldout_range=[args.train_stop, len(stream.tokens)],
                  sampling='fresh epoch-shuffled disjoint contexts; each cached position used once; corpus repeats on exhaustion')
    atomic_json(result, output/'result.json')
    print(json.dumps({'SCALAR_STREAM_COMPLETE': result}), flush=True)
    if args.profile_result is None and not result['complete']:
        raise RuntimeError('Profile time budget exhausted; resume its checkpoint before scheduling training')


if __name__ == '__main__':
    tracking.run_main(main)
