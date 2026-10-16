"""13B-token Matryoshka run on 4 GPUs, on exactly the tokens VPD's published run trained on.

Method: matryoshka67.py's training step, unchanged (imported from it): Fisher K=4 scores, hard top-k forward with the
sigmoid top-k backward, budgets with density k^-1.5 drawn per context and stratified, the lambda = 1 Lp term (p 1 -> 0.5),
and the piece-mass penalty with weight 10. The settings are fixed below; a conflicting environment value is an error.

Data: data/vpd_stream/rank{0..7}.uint16 from prepare_vpd_stream.py = VPD's 8 per-rank streams in update order (as
train_gim_13b.py). VPD global step s = 8 streams x 8 rows. Our step t = VPD steps 4t..4t+3 = 32 rows of each stream,
256 contexts (131,072 positions); GPU g reads streams 2g and 2g+1. 100,000 steps = VPD's 400,000 updates, 13.1B tokens.

Data parallel: every GPU holds identical pieces and runs the step on its 64 contexts (its own budget draws and Fisher
labels); one all-reduce averages the gradients on the free factors (equal shards: the gradient of the mean over all
256 contexts; the mass penalty is the same on every GPU), then every GPU runs the same Adam update and exact-sum
projection. Replicas are compared every 1,000 steps and the run stops if they differ.

Optimizer: the fisher_k4_5000 / gim_1h TrainConfig (Adam lr 1e-3, betas 0.9/0.99) with the cosine to 1e-5 and the
p schedule stretched over the full 100,000 steps (as train_gim_13b.py), and the projection in float64: the fp32
projection let h.1.mlp.down_proj's sum error drift past the 1e-5 guard at step 36,900 of the 13B GIM run (drift_probe.py);
an fp64 projection keeps it at ~1e-8. The guard stays.

No W&B on the compute nodes (no internet): rank 0 appends every logged row to OUT/metrics.jsonl and wandb_relay.py,
running on a login node, forwards new rows to W&B.
Launch: torchrun --nproc_per_node=4 matryoshka13b.py   (SMOKE_STEPS=n: a short run in a separate folder;
SMOKE_ROWS_PER_STREAM=n: read only n rows of each stream per step, for a small GPU). Exit 3 = paused at the time limit (also written to OUT/paused.json with the job id: torchrun reports any
nonzero worker exit as 1).
Checkpoints: the resumable one every 1,000 steps (overwritten), kept copies of it every 10,000, and kept model.pth
snapshots at 1,000 and every 2,500 steps.
"""
import os

SETTINGS = dict(MATRYOSHKA_FORWARD='hard', MATRYOSHKA_SENSOR='fisher', MATRYOSHKA_LP_COEFF='1', MATRYOSHKA_BUDGET_STRATIFIED='1',
                MATRYOSHKA_BUDGET_CONTEXTS='1', MATRYOSHKA_BUDGET_POWER='1.5', MATRYOSHKA_MASS_COEFF='10', MATRYOSHKA_ADV_COEFF='0')
for _key, _value in SETTINGS.items():
    assert os.environ.get(_key, _value) == _value, f'{_key} is fixed to {_value} for this run'
    os.environ[_key] = _value
os.environ['WANDB_MODE'] = 'disabled'                  # never reach for W&B from a compute node

from dataclasses import asdict
from datetime import timedelta
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import matryoshka67 as m
from common import ROOT, MODULES, TARGET_SHA, CONTEXT, credentials, target, tokens, save_json, checkpoint, sha
from train_pair import core, method

STEPS = 100_000
STREAMS, STREAM_ROWS_PER_STEP, ROW = 8, 32, 513
SMOKE = int(os.environ.get('SMOKE_STEPS', '0'))
SMOKE_ROWS = int(os.environ.get('SMOKE_ROWS_PER_STREAM', '0'))
OUT = ROOT/('output/matryoshka13b_smoke' if SMOKE else 'output/matryoshka13b')
CHECK_EVERY = 10 if SMOKE else 1000
CHECKPOINT_EVERY = 10 if SMOKE else 1000
MONITOR_EVERY = 10 if SMOKE else 1000
MILESTONES = {1000} | set(range(2500, STEPS + 1, 2500))     # kept model.pth snapshots (every ~65 min of training)
KEEP_FULL_EVERY = 10_000                                      # kept resumable checkpoints (pieces + Adam + budget states)
EXIT_CONTINUE = 3
# The matched controls' TrainConfig (output/gim_1h/plan.json, as fisher_k4_5000), stretched to this run.
CONTROL_CONFIG = dict(steps=8700, batch=32768, lr=1e-3, lr_min=1e-5, seed=0, log_interval=100, tau=1e-3, p_end=.5, lp_epsilon=1e-6,
                      normalization_events=256, precision='bf16', projection_precision='fp32')
RUN_NAME = 'parauto-matryoshka-13b-vpdstream-hard-lp1-strat-pow1.5-mass10'


def main():
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert STREAMS % world == 0
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    main_proc = rank == 0
    say = (lambda **kw: print(json.dumps(kw), flush=True)) if main_proc else (lambda **kw: None)
    deadline = float(os.environ['RUN_DEADLINE'])
    stop_requested = [False]
    def request_stop(*_): stop_requested[0] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGUSR1, request_stop)

    streams_dir = ROOT/'data/vpd_stream'
    stream_manifest = json.loads((streams_dir/'manifest.json').read_text())
    assert stream_manifest['complete'] and stream_manifest['world_size'] == STREAMS
    assert stream_manifest['batches_per_rank']*stream_manifest['rows_per_rank_batch'] == STEPS*STREAM_ROWS_PER_STEP
    manifest_sha = sha(streams_dir/'manifest.json')
    streams = [np.memmap(streams_dir/f'rank{r}.uint16', mode='r', dtype=np.uint16, shape=(STEPS*STREAM_ROWS_PER_STEP, ROW))
               for r in range(STREAMS)]
    mine = [r for r in range(STREAMS) if r*world//STREAMS == rank]
    per_stream = SMOKE_ROWS or STREAM_ROWS_PER_STEP
    assert per_stream <= STREAM_ROWS_PER_STEP

    def rows(stream_ids, step):
        lo = step*STREAM_ROWS_PER_STEP
        return torch.from_numpy(np.concatenate([np.asarray(streams[r][lo:lo+per_stream], dtype=np.int64) for r in stream_ids])).cuda()

    resuming = (OUT/'latest_checkpoint.json').exists()
    model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    total = sum(m.COUNTS)
    cfg = method.TrainConfig(**{**CONTROL_CONFIG, 'steps': STEPS, 'batch': len(mine)*per_stream*CONTEXT, 'projection_precision': 'fp64'})
    target_steps = SMOKE or STEPS
    weight_norms = [float(w.double().pow(2).sum()) for w in weights]
    weight_mass = sum(weight_norms)
    pieces = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, m.COUNTS)]
    # Lp normalization (matryoshka67): the initial pieces' scores on 256 fixed positions; here the step-0 rows of
    # streams 0 and 1, the same on every GPU.
    first = rows((0, 1), 0)
    index = torch.randperm(first.shape[0]*CONTEXT, device='cuda', generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    reference = m.capture(model, first, seed=123, chunk=m.MICRO)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        initial_scores = [method.structured.scores(method.structured.slice_entry(e, index), p).float() for e, p in zip(reference, pieces)]
    del reference, first
    parameters = [t.requires_grad_() for p in pieces for t in (p.read, p.write)]
    with torch.no_grad():
        for t in parameters: dist.broadcast(t, src=0)
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr, betas=(0.9, 0.99))
    budgets = torch.Generator().manual_seed(cfg.seed + 505 + 1000*rank)
    plan = dict(method='matryoshka67 training step on VPD\'s published-run token stream, 4-GPU data parallel', settings=SETTINGS,
                sensor=m.SENSOR, score=m.SCORES[m.SENSOR], forward=m.FORWARD, k_min=m.K_MIN, budget_power=m.BUDGET_POWER,
                budget_contexts=m.BUDGET_CONTEXTS, budget_stratified=m.BUDGET_STRATIFIED, lp_coeff=m.LP_COEFF, mass_coeff=m.MASS_COEFF,
                training_config=asdict(cfg), lr_note='Adam (0.9, 0.99); cosine 1e-3 -> 1e-5 and p 1 -> 0.5 over the first 80%, both over the full horizon',
                world_size=world, gpus_streams={g: [r for r in range(STREAMS) if r*world//STREAMS == g] for g in range(world)},
                contexts_per_step=STREAMS*per_stream, positions_per_step=STREAMS*per_stream*CONTEXT, steps=target_steps,
                total_positions=target_steps*STREAMS*per_stream*CONTEXT,
                token_source=dict(stream_manifest={k: v for k, v in stream_manifest.items() if k != 'ranks'}, manifest_sha256=manifest_sha,
                                  mapping='our step t = VPD steps 4t..4t+3; VPD step s used per-rank batch s+1'),
                seeds=dict(init=0, lp_reference=123, budgets='seed + 505 + 1000*rank', fisher_labels='1_000_003 + step*world + rank'),
                component_counts=dict(zip(MODULES, m.COUNTS)), target_sha256=TARGET_SHA, smoke=bool(SMOKE), smoke_rows=SMOKE_ROWS,
                monitor=dict(val_rows=[m.MONITOR_ROWS.start, m.MONITOR_ROWS.stop], budgets=list(m.MONITOR_BUDGETS), every=MONITOR_EVERY),
                sum_guard=1e-5, run_name=RUN_NAME, wandb='none on the nodes; OUT/metrics.jsonl relayed by wandb_relay.py')
    start = 0
    if resuming:
        saved_plan = json.loads((OUT/'plan.json').read_text())
        assert {k: v for k, v in saved_plan.items() if k not in ('gpu', 'job_ids')} == json.loads(json.dumps(plan)), 'plan changed'
        full = torch.load(OUT/'checkpoint.pt', map_location='cpu', weights_only=False)
        assert full['target_sha256'] == TARGET_SHA and full['stream_manifest_sha256'] == manifest_sha and full['world_size'] == world
        with torch.no_grad():
            for p, s in zip(pieces, full['state']['pieces'], strict=True):
                p.read.copy_(s['read']); p.write.copy_(s['write'])
        optimizer.load_state_dict(full['state']['optimizer'])
        budgets.set_state(full['state']['budgets'][rank])
        start = full['completed']
        del full
        say(resumed_from=start)
    elif main_proc:
        if (OUT/'plan.json').exists(): say(restarting_from_zero='plan.json exists but no checkpoint; starting fresh')
        OUT.mkdir(parents=True, exist_ok=True)
        save_json({**plan, 'gpu': torch.cuda.get_device_name(), 'job_ids': [os.environ.get('SLURM_JOB_ID')]}, OUT/'plan.json')
        (OUT/'metrics.jsonl').write_text('')
    if resuming and main_proc:
        saved = json.loads((OUT/'plan.json').read_text())
        saved['job_ids'] = saved.get('job_ids', []) + [os.environ.get('SLURM_JOB_ID')]
        save_json(saved, OUT/'plan.json')
    metrics = open(OUT/'metrics.jsonl', 'a') if main_proc else None

    def log(update, **fields):
        if main_proc: metrics.write(json.dumps(dict(update=update, **fields)) + '\n'); metrics.flush()

    val, _ = tokens('val')
    monitor_rows = torch.from_numpy(np.array(val[m.MONITOR_ROWS], dtype=np.int64)).cuda()

    def log_monitor(completed):
        if not main_proc: return
        before = time.perf_counter()
        curve = m.monitor(model, monitor_rows, pieces, seed=77)
        log(completed, **{f'monitor/kl_keep_{k}': v for k, v in curve.items()}, **{'monitor/seconds': time.perf_counter() - before})
        say(update=completed, monitor=curve)

    def save(completed):
        states = [None]*world
        dist.all_gather_object(states, budgets.get_state())
        if not main_proc: return
        before = time.perf_counter()
        identity = dict(method='parauto', sensor=m.SENSOR, completed=completed, training_positions=completed*plan['positions_per_step'],
                        target_sha256=TARGET_SHA, stream_manifest_sha256=manifest_sha, world_size=world)
        factors = [{'read': p.read.detach(), 'write': p.write.detach()} for p in pieces]
        full = dict(**identity, state=dict(pieces=factors, optimizer=optimizer.state_dict(), budgets=states))
        checkpoint(full, OUT/'checkpoint.pt')
        if completed % KEEP_FULL_EVERY == 0 and not SMOKE: checkpoint(full, OUT/'checkpoints'/f'{completed:06d}'/'checkpoint.pt')
        if completed in MILESTONES or completed == target_steps:
            checkpoint(dict(**identity, factors=factors, provenance=dict(kind='Matryoshka 13B, VPD stream, 4-GPU data parallel', plan=plan)),
                       OUT/'checkpoints'/f'{completed:06d}'/'model.pth')
        save_json(dict(step=completed, seconds=time.perf_counter() - before), OUT/'latest_checkpoint.json')
        say(checkpoint=completed)

    def replicas_agree(completed):
        digest = torch.stack([torch.stack((p.read.double().sum(), p.write.double().sum(), p.read.double().square().sum())) for p in pieces]).flatten()
        gathered = [torch.zeros_like(digest) for _ in range(world)]
        dist.all_gather(gathered, digest)
        diff = max(float((g - gathered[0]).abs().max()) for g in gathered)
        scale = float(gathered[0].abs().max())
        log(completed, **{'check/replica_max_abs_diff': diff})
        if diff > 1e-9*scale: raise RuntimeError(f'replicas diverged at step {completed}: {diff}')

    if start == 0: log_monitor(0)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    completed = start
    for step in range(start, target_steps):
        tick = time.perf_counter()
        for group in optimizer.param_groups: group['lr'] = core.cosine_lr(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        batch_rows = rows(mine, step)
        cache, target_logp = m.capture(model, batch_rows, seed=1_000_003 + step*world + rank, keep_logp=True)
        feasible = [method.project(w, p, cfg) for w, p in zip(weights, pieces)]
        draws = m.update_budgets(budgets, total, len(batch_rows) // m.BUDGET_CONTEXTS)
        p_now = 1 + (cfg.p_end - 1) * min(1.0, step / max(1, cfg.steps * 0.8))
        lp = dict(coeff=m.LP_COEFF, p=p_now, epsilon=cfg.lp_epsilon,
                  denominators=[method.lp_concentration(s, p_now, cfg.lp_epsilon).mean().clamp_min(1e-8) for s in initial_scores])
        leaves, stats = m.accumulate(model, batch_rows, cache, feasible, draws, lp=lp, target_logp=target_logp)
        del cache, target_logp
        outputs, cotangents = [t for f in feasible for t in (f.read, f.write)], [t.grad for l in leaves for t in (l.read, l.write)]
        masses = m.piece_mass(feasible, weight_norms)
        outputs.append(m.MASS_COEFF * (masses - 1).pow(2).mean()); cotangents.append(torch.ones((), device=masses.device))
        torch.autograd.backward(outputs, cotangents)
        del leaves, feasible
        grads = [t.grad for t in parameters]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads)): grad.copy_(synced)
        del flat
        grad_norm = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))
        if not math.isfinite(grad_norm): raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        for w, p in zip(weights, pieces): method.project_(w, p, cfg)
        completed = step + 1
        local = torch.tensor([np.mean([s['kl'] for s in stats]), np.mean([s['on_over_k'] for s in stats]),
                              np.mean([s['soft_leak'] for s in stats]), np.mean([s['lp_normalized'] for s in stats]),
                              1. if (stop_requested[0] or time.time() >= deadline) else 0.], device='cuda', dtype=torch.float64)
        flag = local[4:].clone(); dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        means = local[:4].clone(); dist.all_reduce(means, op=dist.ReduceOp.AVG)
        kl, on_over_k, soft_leak, lp_normalized = means.tolist()
        m_ = masses.detach()
        row = dict(kl=kl, lr=core.cosine_lr(step, cfg), grad_norm=grad_norm, on_over_k=on_over_k, soft_leak=soft_leak,
                   lp_normalized=lp_normalized, p=p_now, mass_penalty=float((m_ - 1).pow(2).mean()), mass_max=float(m_.max()),
                   mass_total=float((m_ * torch.tensor(weight_norms, device=m_.device)).sum()) / weight_mass,
                   seconds_this_step=time.perf_counter() - tick, seconds_per_step=(time.perf_counter() - started) / (completed - start),
                   peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, positions=completed*plan['positions_per_step'],
                   remaining_hours=max(0., deadline - time.time()) / 3600)
        for lo, hi, label in ((0, 100, 'lt100'), (100, 1000, '100_1000'), (1000, 10000, '1000_10000'), (10000, total, 'ge10000')):
            inside = [s['kl'] for s in stats if lo <= s['k'] < hi]       # this GPU's contexts
            if inside: row[f'kl_k_{label}'] = float(np.mean(inside))
        if not all(math.isfinite(v) for v in row.values()): raise RuntimeError(f'Nonfinite metrics: {row}')
        log(completed, **{'train/'+k: v for k, v in row.items()})
        if completed % 10 == 0 or completed <= 5: say(update=completed, **row, budgets_gpu0=draws)
        if completed % 100 == 0 or completed == target_steps:
            errors = [core.reconstruction_error(w, p) for w, p in zip(weights, pieces)]
            log(completed, **{'train/max_sum_error': max(errors)})
            if max(errors) > 1e-5: raise RuntimeError(f'Matrix sum drift: {max(errors)}')
            if main_proc: save_json(dict(update=completed, max_sum_error=max(errors), **row), OUT/'progress.json')
        if completed % CHECK_EVERY == 0: replicas_agree(completed)
        if completed % MONITOR_EVERY == 0 or completed == target_steps: log_monitor(completed)
        stop = bool(flag.item())
        if completed % CHECKPOINT_EVERY == 0 or (completed in MILESTONES and not SMOKE) or completed == target_steps or stop: save(completed)
        if stop and completed < target_steps:
            reason = 'signal' if stop_requested[0] else 'time_budget'
            if main_proc: save_json(dict(completed=completed, job=os.environ.get('SLURM_JOB_ID'), reason=reason), OUT/'paused.json')
            say(paused_at=completed, reason=reason)
            dist.barrier(); dist.destroy_process_group()
            sys.exit(EXIT_CONTINUE)
    if main_proc:
        result = dict(completed=completed, positions=completed*plan['positions_per_step'], resumed_from=start,
                      seconds_this_job=time.perf_counter() - started, peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
        save_json(result, OUT/'result.json'); say(finished=result)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__': main()
