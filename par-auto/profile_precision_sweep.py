"""Isolated precision ablations; defaults and historical training stay intact."""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import parauto_core as core
import parauto_method as method
import parauto_structured as structured
import parauto_toy as toy
from parauto_target import atomic_json, atomic_save, fingerprint, load_target, sample
from parauto_evaluate import TARGET_SHA


class HalfGradientScaler:
    """Ordinary dynamic scaling without FP32 masters; native AMP forbids FP16 grads.

    Unscale in place in the gradient's own dtype. Keep stock Adam and its
    original epsilon. This deliberately does not repair FP16 Adam underflow.
    """
    def __init__(self, *args, **kwargs):
        self.value, self.good, self.bad = 65536., 0, False

    def is_enabled(self): return True
    def get_scale(self): return self.value
    def scale(self, loss): return loss * self.value

    def step(self, optimizer):
        grads = [p.grad for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
        self.bad = not bool(torch.stack([g.isfinite().all() for g in grads]).all())
        if not self.bad:
            for g in grads: g.mul_(1 / self.value)
            optimizer.step()

    def update(self):
        if self.bad:
            self.value *= .5; self.good = 0
        else:
            self.good += 1
            if self.good == 2000:
                self.value *= 2; self.good = 0


def projected_fp32(weight, pieces):
    """Same affine projection and autograd, changing only FP64 to FP32."""
    reads, writes, w = pieces.read.float(), pieces.write.float(), weight.float()
    residual = w - writes @ reads.T
    if w.shape[0] <= w.shape[1]:
        reads = reads + residual.T @ torch.linalg.solve(writes @ writes.T, writes)
        return core.FreePieces(reads.to(pieces.read.dtype), pieces.write)
    writes = writes + residual @ torch.linalg.solve(reads @ reads.T, reads)
    return core.FreePieces(pieces.read, writes.to(pieces.write.dtype))


def half_cache(cache):
    """Cast cached floating activations/maps; retain diagnostic net effects in FP32."""
    seen = {}
    def visit(value, key=''):
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.float32 or key == 'net_rms': return value
            if id(value) not in seen: seen[id(value)] = value.half()
            return seen[id(value)]
        if isinstance(value, dict): return {k: visit(v, k) for k, v in value.items()}
        return value
    return visit(cache)


def float_entry(entry):
    return {**entry, 'h': entry['h'].float(),
            'shared': {k: v.float() if v.is_floating_point() else v for k, v in entry['shared'].items()},
            'events': {k: v.float() if v.is_floating_point() else v for k, v in entry['events'].items()}}


def half_intermediate_sensitivity(entry, write):
    """Keep large score intermediates in FP16, using ordinary vector norms.

    Only the reduced [batch, components] norms are promoted to FP32. Audit
    calls retain the original FP32 arithmetic outside the autocast context.
    """
    shared, events = entry['shared'], entry['events']
    def reduce(value, dim):
        return torch.linalg.vector_norm(value, dim=dim).float().square()
    kind = entry['kind']
    if kind == 'input_qr': return reduce(events['factor'] @ write, -2)
    if kind == 'input_mask':
        factor = shared['factor'].unsqueeze(0) * events['mask'].unsqueeze(1)
        return reduce(factor @ write, -2)
    if kind == 'output_shared': return reduce(shared['output'] @ write, 0)
    if kind == 'output_residual':
        base = shared['output'] @ write
        rank = shared['response'].shape[0]
        hidden = events['mask'].unsqueeze(-1) * (shared['read'] @ write)
        effect = shared['response'] @ hidden + base[:rank]
        return reduce(effect, -2) + reduce(base[rank:], 0)
    raise ValueError(kind)


def tensor_summary(tensors):
    by_dtype = {}
    for t in tensors:
        key = str(t.dtype)
        row = by_dtype.setdefault(key, {'tensors': 0, 'bytes': 0})
        row['tensors'] += 1; row['bytes'] += t.numel() * t.element_size()
    return by_dtype


import parauto_tracking as tracking

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('baseline', 'parameters_fp16', 'activations_fp16', 'projection_fp32'), required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--activation-scope', choices=('cache', 'cache_and_intermediates'), default='cache_and_intermediates')
    args = parser.parse_args()
    tracking.start_cli(args)
    assert fingerprint(args.target) == TARGET_SHA
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    target = load_target(args.target, 'cuda')
    cfg = method.TrainConfig(steps=args.steps, batch=2048, precision='fp16', log_interval=500)
    cache = structured.cache_events(target, sample(16384, 100, torch.Generator(device='cuda').manual_seed(101), 'cuda'))
    named = toy.matrices(target); weights = [w for _, w in named]
    entries = [cache['matrices'][name] for name, _ in named]
    initial = [method.prepare_initial(w, 400, cfg, log=lambda row: None)[0] for e, w in zip(entries, weights)]
    result = {'variant': args.variant, 'requested_steps': args.steps, 'seed': 0, 'target_sha256': TARGET_SHA,
              'target_steps': 10000, 'batch': 2048, 'components_per_matrix': 400,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'status': 'started',
              'source_sha256': fingerprint(__file__), 'baseline': 'FP16 score matmuls; FP32 parameters/Adam/cache/normalization/norms; FP64 projection'}
    if args.variant == 'parameters_fp16':
        initial = [core.FreePieces(p.read.half(), p.write.half()) for p in initial]
        torch.amp.GradScaler = HalfGradientScaler
        result['scaler_note'] = 'Manual dynamic in-place FP16 gradient unscale; stock AMP rejects FP16 gradient storage. Adam epsilon stays 1e-8; no FP32 masters.'
    elif args.variant == 'activations_fp16':
        cache = half_cache(cache)
        entries = [cache['matrices'][name] for name, _ in named]
        result['activation_scope'] = args.activation_scope
        if args.activation_scope == 'cache_and_intermediates':
            original_sensitivity = structured.squared_sensitivity
            def sensitivity(entry, write):
                if torch.is_autocast_enabled('cuda'):
                    return half_intermediate_sensitivity(entry, write)
                return original_sensitivity(entry, write)
            structured.squared_sensitivity = sensitivity
    elif args.variant == 'projection_fp32':
        core.projected_pieces = projected_fp32
    # Reference normalization scores retain FP32 arithmetic in both storage
    # ablations to avoid mixed-dtype matmul failures.
    if args.variant in ('parameters_fp16', 'activations_fp16'):
        original_scores = method.scores_of
        def normalization_compatible_scores(entry, pieces):
            if torch.is_autocast_enabled('cuda'): return original_scores(entry, pieces)
            return original_scores(float_entry(entry), core.FreePieces(pieces.read.float(), pieces.write.float()))
        method.scores_of = normalization_compatible_scores
    result['cache_bytes'] = structured.cache_bytes(cache)
    result['initial_sum_errors'] = {name: core.reconstruction_error(w, p) for (name, w), p in zip(named, initial)}
    result['initial_parameter_storage'] = tensor_summary([t for p in initial for t in (p.read, p.write)])
    atomic_json(result, args.output / 'result.json')
    original_step, original_boundary = torch.optim.Adam.step, method.optimizer_step
    captured, counters = {}, {'scheduled_steps': 0, 'adam_updates': 0}
    warm_start = None

    def adam_step(optimizer, *a, **kw):
        captured['optimizer'] = optimizer
        value = original_step(optimizer, *a, **kw)
        counters['adam_updates'] += 1
        if args.variant == 'parameters_fp16':
            params = [p for g in optimizer.param_groups for p in g['params']]
            if not bool(torch.stack([p.isfinite().all() for p in params]).all()):
                raise RuntimeError('Stock Adam produced nonfinite FP16 parameters; unchanged epsilon=1e-8 and FP16 moments')
        return value

    def boundary(optimizer, scaler):
        nonlocal warm_start
        captured['optimizer'] = optimizer
        value = original_boundary(optimizer, scaler)
        counters['scheduled_steps'] += 1
        if counters['scheduled_steps'] == 200:
            torch.cuda.synchronize(); warm_start = time.perf_counter()
        return value

    torch.optim.Adam.step, method.optimizer_step = adam_step, boundary
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    try:
        fitted, histories = method.fit(entries, weights, initial, cfg, log=lambda row: print(json.dumps(row), flush=True))
        torch.cuda.synchronize(); finished = time.perf_counter()
        result.update(status='completed', overflow_skipped_updates=histories[0][-1]['overflow_skipped_updates'],
                      final_sum_errors={name: core.reconstruction_error(w, p) for (name, w), p in zip(named, fitted)},
                      optimizer_updates=[rows[-1]['optimizer_updates'] for rows in histories])
        atomic_json(histories, args.output / 'training.json')
        atomic_save({'pieces': [{'read': p.read.cpu(), 'write': p.write.cpu()} for p in fitted]}, args.output / 'pieces.pt')
    except Exception as exc:
        torch.cuda.synchronize(); finished = time.perf_counter()
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    result.update(counters, fit_seconds=finished-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if result['status'] == 'completed':
        result['ms_per_scheduled_step'] = result['fit_seconds'] * 1000 / args.steps
        result['warm_ms_per_step'] = (finished-warm_start) * 1000 / (args.steps-200)
    if 'optimizer' in captured:
        opt = captured['optimizer']; params = [p for g in opt.param_groups for p in g['params']]
        result['parameter_storage'] = tensor_summary(params)
        result['gradient_storage'] = tensor_summary([p.grad for p in params if p.grad is not None])
        result['adam_moment_storage'] = tensor_summary([v for state in opt.state.values() for k, v in state.items() if k in ('exp_avg', 'exp_avg_sq')])
    atomic_json(result, args.output / 'result.json')
    print(json.dumps({'PROFILE_COMPLETE': args.variant, 'result': result}), flush=True)


if __name__ == '__main__': tracking.run_main(main)
