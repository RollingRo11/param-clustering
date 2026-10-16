"""Where does one Matryoshka training step spend its time?  lambda = 1, hard forward, Fisher score (main_hard_lp1).

Timing only; no parameters or checkpoints are written.  Starting from the lambda = 1 run's 5,000-step pieces:
  1. the training step (matryoshka67.main's loop body without W&B, monitor or checkpoints), averaged, for several
     (masked-KL chunk, score-cache chunk) sizes in contexts; (4, 4) is the old batching with the new kernels;
  2. at MATRYOSHKA_CHUNK_CONTEXTS, the same step with a synchronized timer around each stage (the syncs add a little);
  3. the top GPU ops of one step (torch.profiler) and the GPU-busy fraction of that step;
  4. the exact-sum projection of all 24 matrices in fp32 (every run so far) and in fp64.
The pre-change step measured 1.80 s on an H200 (job 10646857, commit 3c2ccc6).
Usage: profile_matryoshka67.py OUTPUT_JSON   (MATRYOSHKA_FORWARD=hard MATRYOSHKA_LP_COEFF=1 set by the job script)
"""
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

import matryoshka67 as m

assert m.FORWARD == 'hard' and m.LP_COEFF == 1 and m.SENSOR == 'fisher'
WARMUP, PLAIN, TIMED = 2, 5, 3
CONFIGS = ((4, 4), (16, 4), (16, 16), (32, 32))      # (masked-KL chunk, score-cache chunk), contexts


class Stages:
    def __init__(self): self.seconds = defaultdict(float)

    @contextmanager
    def __call__(self, name):
        torch.cuda.synchronize(); start = time.perf_counter()
        yield
        torch.cuda.synchronize(); self.seconds[name] += time.perf_counter() - start


def main():
    output = Path(sys.argv[1])
    m.credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.manual_seed(0)
    data, _ = m.tokens(); model = m.target()
    weights = [model.get_submodule(n).weight for n in m.MODULES]
    total = sum(m.COUNTS)
    cfg = m.method.TrainConfig(**json.loads((m.ROOT/'output/gim_1h/plan.json').read_text())['training_config'])
    # Lp denominators exactly as main(): initial pieces on the fixed 256-position reference set.
    initial = [m.core.initialize_pieces(w, c, 0) for w, c in zip(weights, m.COUNTS)]
    index = torch.randperm(m.BATCH*m.CONTEXT, device='cuda', generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    first = m.capture(model, m.batch(data, 0), seed=123, chunk=m.MICRO)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        initial_scores = [m.method.structured.scores(m.method.structured.slice_entry(e, index), p).float() for e, p in zip(first, initial)]
    del first, initial
    saved = torch.load(m.ROOT/'output/matryoshka67_20260923/main_hard_lp1/checkpoints/005000/model.pth', map_location='cuda', weights_only=False)
    pieces = [m.core.FreePieces(f['read'].clone(), f['write'].clone()) for f in saved['factors']]
    del saved
    parameters = [t.requires_grad_() for p in pieces for t in (p.read, p.write)]
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr, betas=(0.9, 0.99))
    budgets = torch.Generator().manual_seed(cfg.seed + 505)

    def lp_at(step):
        p_now = 1 + (cfg.p_end - 1) * min(1.0, step / max(1, cfg.steps * 0.8))
        return dict(coeff=m.LP_COEFF, p=p_now, epsilon=cfg.lp_epsilon,
                    denominators=[m.method.lp_concentration(s, p_now, cfg.lp_epsilon).mean().clamp_min(1e-8) for s in initial_scores])

    def plain_step(step, chunk, capture_chunk):
        """matryoshka67.main's step, minus W&B logging, the monitor and checkpoints."""
        for group in optimizer.param_groups: group['lr'] = m.core.cosine_lr(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        rows = m.batch(data, step)
        cache, target_logp = m.capture(model, rows, seed=1_000_003+step, chunk=capture_chunk, keep_logp=True)
        feasible = [m.method.project(w, p, cfg) for w, p in zip(weights, pieces)]
        draws = [m.budget(budgets, total) for _ in range(m.BATCH // m.MICRO)]
        leaves, stats = m.accumulate(model, rows, cache, feasible, draws, lp=lp_at(step), target_logp=target_logp, chunk=chunk)
        del cache, target_logp
        torch.autograd.backward([t for f in feasible for t in (f.read, f.write)], [t.grad for l in leaves for t in (l.read, l.write)])
        del leaves, feasible
        grad_norm = float(torch.linalg.vector_norm(torch.stack([t.grad.norm() for t in parameters])))
        assert math.isfinite(grad_norm) and all(math.isfinite(s['kl']) for s in stats)
        optimizer.step()
        for w, p in zip(weights, pieces): m.method.project_(w, p, cfg)
        torch.cuda.synchronize()

    def timed_step(step, T, chunk):
        """The same step, with accumulate() written out so each stage gets its own timer."""
        with T('other (lr, zero_grad, batch, budgets, Lp denominators)'):
            for group in optimizer.param_groups: group['lr'] = m.core.cosine_lr(step, cfg)
            optimizer.zero_grad(set_to_none=True)
            rows = m.batch(data, step)
            draws = [m.budget(budgets, total) for _ in range(m.BATCH // m.MICRO)]
            lp = lp_at(step)
        with T('score cache + intact log-probs (capture_fisher)'):
            cache, target_logp = m.capture(model, rows, seed=1_000_003+step, chunk=chunk, keep_logp=True)
        with T('exact-sum projection, in graph (before the step)'):
            feasible = [m.method.project(w, p, cfg) for w, p in zip(weights, pieces)]
        leaves = [m.core.FreePieces(f.read.detach().requires_grad_(), f.write.detach().requires_grad_()) for f in feasible]
        group = m.MICRO*m.CONTEXT
        pending = []
        for first in range(0, len(rows), chunk):
            block = rows[first:first+chunk]; ids = block[:, :-1]
            lo, hi = first*m.CONTEXT, (first+len(block))*m.CONTEXT
            ks = draws[first//m.MICRO:(first+len(block))//m.MICRO]
            target = target_logp[lo:hi].view(*ids.shape, -1)
            parts = []
            with T('scores of all 43,008 pieces + log'):
                s = m.log_scores(cache, leaves, lo, hi, parts_out=parts)
            with T('bisection for the soft top-k shift (Triton)'):
                c = m.threshold_shift(s.detach(), torch.tensor(ks, dtype=s.dtype, device=s.device).repeat_interleave(group))
            with T('soft mask + hard threshold (topk)'):
                soft = m.SigmoidTopk.apply(s, c)
                with torch.no_grad():
                    threshold = torch.cat([s.detach()[i*group:(i+1)*group].topk(k, dim=-1, sorted=False).values.amin(-1, keepdim=True)
                                           for i, k in enumerate(ks)])
                mask = m.HardTopk.apply(soft, s.detach(), threshold)
            alphas = [a.reshape(*ids.shape, -1) for a in mask.split(m.COUNTS, -1)]
            with T('masked forward + KL'):
                kl = m.kl_per_position(target, m.masked_logits(model, ids, leaves, alphas)).view(len(ks), -1).mean(-1)
            with T('Lp term'):
                term = m.lp_term(parts, lp)
            with T('backward (masked forward, scores, top-k)'):
                ((kl.sum() + lp['coeff'] * term * len(ks)) / len(draws)).backward()
            with T('statistics (on device)'), torch.no_grad():
                on = soft > .5
                pending.append((kl.detach(), on.sum(-1).float().view(len(ks), -1).mean(-1), (soft * ~on).sum(-1).view(len(ks), -1).mean(-1)))
            del target, s, c, soft, threshold, mask, alphas, kl, on, parts, term
        with T('statistics (one host sync)'):
            _ = [[t.tolist() for t in row] for row in pending]
        del cache, target_logp
        with T('backward through the in-graph projection'):
            torch.autograd.backward([t for f in feasible for t in (f.read, f.write)], [t.grad for l in leaves for t in (l.read, l.write)])
        del leaves, feasible
        with T('grad norm + Adam'):
            grad_norm = float(torch.linalg.vector_norm(torch.stack([t.grad.norm() for t in parameters])))
            optimizer.step()
        with T('exact-sum projection after Adam'):
            for w, p in zip(weights, pieces): m.method.project_(w, p, cfg)

    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, config='main_hard_lp1 from its step-5000 pieces',
                  positions_per_step=m.BATCH*m.CONTEXT, budget_draws_per_step=m.BATCH // m.MICRO, before_seconds_per_step=1.8049,
                  plain={})
    step = 5000
    for chunk, capture_chunk in CONFIGS:
        for _ in range(WARMUP): plain_step(step, chunk, capture_chunk); step += 1
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(PLAIN): plain_step(step, chunk, capture_chunk); step += 1
        seconds = (time.perf_counter() - start) / PLAIN
        row = dict(seconds_per_step=seconds, hours_for_1b_tokens=seconds * math.ceil(1e9 / (m.BATCH*m.CONTEXT)) / 3600,
                   peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
        result['plain'][f'kl_chunk_{chunk}_cache_chunk_{capture_chunk}'] = row
        print(json.dumps(dict(kl_chunk=chunk, cache_chunk=capture_chunk, **row)), flush=True)

    T = Stages()
    start = time.perf_counter()
    for _ in range(TIMED): timed_step(step, T, m.CHUNK); step += 1
    result['timed_chunk'] = m.CHUNK
    result['timed_seconds_per_step'] = (time.perf_counter() - start) / TIMED
    result['stages_seconds_per_step'] = {k: v / TIMED for k, v in sorted(T.seconds.items(), key=lambda kv: -kv[1])}
    print(json.dumps(dict(timed_seconds_per_step=result['timed_seconds_per_step'], stages=result['stages_seconds_per_step']), indent=1), flush=True)

    t0 = time.perf_counter()
    m.monitor(model, torch.from_numpy(np.array(m.tokens('val')[0][m.MONITOR_ROWS], dtype=np.int64)).cuda(), pieces, seed=77)
    torch.cuda.synchronize(); result['monitor_seconds_once'] = time.perf_counter() - t0

    from torch.autograd import DeviceType
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize(); wall = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        plain_step(step, m.CHUNK, m.CHUNK); step += 1
    wall = time.perf_counter() - wall
    averages = prof.key_averages()
    def device_us(e): return getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0)
    kernels = [e for e in averages if e.device_type == DeviceType.CUDA]
    ops = [e for e in averages if e.device_type == DeviceType.CPU and device_us(e) > 0]     # each kernel counted once, under its op
    result['profiled_step_wall_seconds'] = wall
    result['profiled_step_gpu_kernel_seconds'] = sum(device_us(e) for e in kernels) / 1e6
    result['profiled_step_kernel_launches'] = sum(e.count for e in kernels)
    result['top_ops'] = [dict(op=e.key[:100], gpu_seconds=device_us(e) / 1e6, calls=e.count)
                         for e in sorted(ops, key=device_us, reverse=True)[:25]]
    print(json.dumps({k: result[k] for k in ('profiled_step_wall_seconds', 'profiled_step_gpu_kernel_seconds', 'profiled_step_kernel_launches')}), flush=True)
    for row in result['top_ops']: print(json.dumps(row), flush=True)
    del prof, averages, kernels, ops

    projection = {}
    for precision in ('fp32', 'fp64'):
        precise = replace(cfg, projection_precision=precision)
        with torch.no_grad():
            for w, p in zip(weights, pieces): m.method.project_(w, p, precise)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(3):
                for w, p in zip(weights, pieces): m.method.project_(w, p, precise)
            torch.cuda.synchronize()
        projection[precision] = (time.perf_counter() - t0) / 3
    result['projection_after_adam_seconds'] = projection
    print(json.dumps(dict(projection_after_adam_seconds=projection)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))
    print('PROFILE_DONE', output, flush=True)


if __name__ == '__main__': main()
