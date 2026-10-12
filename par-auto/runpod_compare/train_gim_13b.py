"""13B-token par-auto GIM run on 4 GPUs, on exactly the tokens VPD's published run trained on.

Launch: torchrun --nproc_per_node=4 train_gim_13b.py [--resume]

Data: data/vpd_stream/rank{0..7}.uint16 from prepare_vpd_stream.py = VPD's 8 per-rank streams, update order.
VPD global step s = 8 ranks x 8 rows. Our step t = VPD steps 4t..4t+3 = 32 rows of each of the 8 streams,
256 rows (131,072 positions). GPU g reads streams 2g and 2g+1. 100,000 steps = VPD's 400,000 updates.

Data parallel: every GPU holds identical pieces and computes the GIM objective on its 64 sequences; one
all-reduce averages the gradients (equal shards, so this is the gradient of the mean over all 131,072
positions), then every GPU runs the same Adam update and exact-sum projection. Normalizers come from the
same reference rows on every GPU and are broadcast from rank 0. Replicas are compared every 1,000 steps.

Config: the earlier par-auto GIM runs' TrainConfig at this scale - 32,768 positions per GPU, LR 1e-3
cosine to 1e-5 and p 1 -> 0.5 over the first 80%, both over the full 100,000-step horizon; bf16 scores,
fp32 projection; GIM backward rules as in gim_attribution.
"""
import json
import os
import signal
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from common import ROOT, MODULES, TARGET_SHA, credentials, target, save_json, checkpoint, sha
from train_pair import core, method
import parauto_tracking as tracking
from gim_attribution import capture_gim, RULES

STEPS = 100_000
STREAMS, STREAM_ROWS_PER_STEP, ROW, CONTEXT = 8, 32, 513, 512
SMOKE = int(os.environ.get('SMOKE_STEPS', '0'))                       # >0: short end-to-end check, separate folder
OUT = ROOT/('output/gim_13b_smoke' if SMOKE else 'output/gim_13b')
CHECK_EVERY = 10 if SMOKE else 1000
MILESTONES = {1000, 5000, 10000, 25000, 50000, 75000, STEPS}
COUNTS = dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)


def main():
    resuming = '--resume' in sys.argv
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, 'the token mapping (2 VPD streams per GPU) assumes 4 GPUs'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0)
    main_proc = rank == 0
    if not main_proc: tracking.log_train = lambda *_a, **_k: None
    say = (lambda **kw: print(json.dumps(kw), flush=True)) if main_proc else (lambda **kw: None)
    deadline = float(os.environ['RUN_DEADLINE'])
    stop_requested = [False]                                                 # set in the handler, acted on per step
    def request_stop(*_): stop_requested[0] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGUSR1, request_stop)

    streams_dir = ROOT/'data/vpd_stream'
    stream_manifest = json.loads((streams_dir/'manifest.json').read_text())
    assert stream_manifest['complete'] and stream_manifest['world_size'] == STREAMS
    assert stream_manifest['batches_per_rank']*stream_manifest['rows_per_rank_batch'] == STEPS*STREAM_ROWS_PER_STEP
    manifest_sha = sha(streams_dir/'manifest.json')
    streams = [np.memmap(streams_dir/f'rank{r}.uint16', mode='r', dtype=np.uint16, shape=(STEPS*STREAM_ROWS_PER_STEP, ROW))
               for r in range(STREAMS)]
    mine = [r for r in range(STREAMS) if r*world//STREAMS == rank]            # GPU g reads streams 2g, 2g+1

    def rows(stream_ids, step):
        lo = step*STREAM_ROWS_PER_STEP
        return torch.from_numpy(np.concatenate([np.asarray(streams[r][lo:lo+STREAM_ROWS_PER_STEP], dtype=np.int64)
                                                for r in stream_ids])).cuda()

    exists = (OUT/'plan.json').exists()
    # A job that died before its first checkpoint leaves plan.json with nothing to resume from.
    # Treat that as a fresh start rather than failing, and say so loudly.
    if resuming and exists and not (OUT/'checkpoint.pt').exists():
        say(restarting_from_zero='plan.json exists but no checkpoint was written; starting fresh')
        resuming = False
    elif exists != resuming:
        raise FileExistsError('Use --resume exactly when this run already exists')
    model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    counts = [COUNTS[n.split('.')[-1]] for n in MODULES]
    cfg = method.TrainConfig(steps=STEPS, batch=len(mine)*STREAM_ROWS_PER_STEP*CONTEXT, lr=1e-3, lr_min=1e-5, p_end=.5,
                             log_interval=100, precision='bf16', projection_precision='fp32')
    initial = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, counts)]
    # Same reference rows on every GPU (global step 0, streams 0 and 1), same 256 positions as earlier runs.
    index = torch.randperm(cfg.batch, device='cuda', generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    reference = [method.slice_entry(e, index) for e in capture_gim(model, rows((0, 1), 0))]

    resume, start = None, 0
    if resuming:
        full = torch.load(OUT/'checkpoint.pt', map_location='cpu', weights_only=False)
        assert full['target_sha256'] == TARGET_SHA and full['stream_manifest_sha256'] == manifest_sha
        resume, start = full['state'], full['completed']
        if main_proc:
            torch.set_rng_state(full['torch_rng'])
        del full
        say(resumed_from=start)
    plan = dict(sensor='gim', rules=RULES, training_config=asdict(cfg), world_size=world, gpus_streams={g: [r for r in range(STREAMS) if r*world//STREAMS == g] for g in range(world)},
                positions_per_step=STREAMS*STREAM_ROWS_PER_STEP*CONTEXT, total_positions=STEPS*STREAMS*STREAM_ROWS_PER_STEP*CONTEXT,
                token_source=dict(stream_manifest={k: v for k, v in stream_manifest.items() if k != 'ranks'}, manifest_sha256=manifest_sha,
                                  mapping='our step t = VPD steps 4t..4t+3; VPD step s used per-rank batch s+1'),
                component_counts=dict(zip(MODULES, counts)), reference_index_seed=123, target_sha256=TARGET_SHA,
                gpu=torch.cuda.get_device_name(), job_id=os.environ.get('SLURM_JOB_ID'))
    run = None
    if main_proc:
        if SMOKE: os.environ['WANDB_MODE'] = 'disabled'
        # These compute nodes have no internet: log to disk and `wandb sync` from a login node later.
        os.environ.setdefault('WANDB_MODE', 'offline')
        import wandb
        if not resuming:
            OUT.mkdir(parents=True, exist_ok=True); save_json(plan, OUT/'plan.json')
        previous = json.loads((OUT/'wandb.json').read_text()) if resuming and (OUT/'wandb.json').exists() else None
        run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu', group='gim-13b-vpd-stream',
                         name='parauto-gim-13b-vpdstream', job_type='training', dir=str(OUT), config=plan,
                         id=previous['id'] if previous else None, resume='must' if previous else None)
        save_json(dict(id=run.id, url=run.url), OUT/'wandb.json')
        run.define_metric('update'); run.define_metric('train/*', step_metric='update')

    def after_backward(parameters):
        grads = [p.grad for p in parameters]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads)): grad.copy_(synced)

    def sync_initial(tensors):
        for t in tensors: dist.broadcast(t, src=0)

    def after_step(next_step, pieces):
        if next_step % CHECK_EVERY: return
        digest = torch.stack([torch.stack((p.read.double().sum(), p.write.double().sum(), p.read.double().square().sum()))
                              for p in pieces]).flatten()
        gathered = [torch.zeros_like(digest) for _ in range(world)]
        dist.all_gather(gathered, digest)
        diff = max(float((g-gathered[0]).abs().max()) for g in gathered)
        scale = float(gathered[0].abs().max())
        say(replica_check=dict(step=next_step, max_abs_diff=diff, relative=diff/max(scale, 1e-30)))
        if diff > 1e-9*scale: raise RuntimeError(f'replicas diverged at step {next_step}: {diff}')

    started, completed = time.perf_counter(), start
    def progress(next_step):
        nonlocal completed
        completed = next_step
        local = stop_requested[0] or time.time() >= deadline or bool(SMOKE and next_step >= SMOKE)
        flag = torch.tensor([1. if local else 0.], device='cuda')
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)                          # all GPUs stop at the same step
        if next_step % (5 if SMOKE else 100) == 0 and main_proc:
            torch.cuda.synchronize()
            row = dict(update=next_step, positions=next_step*plan['positions_per_step'],
                       seconds_per_step=(time.perf_counter()-started)/max(1, next_step-start),
                       peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9, remaining_seconds=max(0., deadline-time.time()))
            save_json(row, OUT/'progress.json'); run.log({'update': next_step, **{'train/'+k: v for k, v in row.items() if k != 'update'}})
            say(**row)
        return bool(flag.item())

    def save(state):
        if not main_proc: return
        step = state['next_step']
        identity = dict(method='parauto', sensor='gim', completed=step, training_positions=step*plan['positions_per_step'],
                        target_sha256=TARGET_SHA, stream_manifest_sha256=manifest_sha)
        checkpoint(dict(**identity, state=state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), OUT/'checkpoint.pt')
        if step in MILESTONES and not SMOKE:
            checkpoint(dict(**identity, factors=state['pieces'], provenance=dict(kind='13B VPD-stream GIM, 4-GPU data parallel',
                            rules=RULES, training_config=asdict(cfg))), OUT/'checkpoints'/f'{step:06d}'/'model.pth')
        save_json(dict(step=step), OUT/'latest_checkpoint.json')
        say(checkpoint=step)

    def log(row):
        if main_proc and row['matrix_index'] == 0:
            run.log({'update': row['step']+1, 'train/total_loss': row['total_loss'], 'train/p': row['p'],
                     'train/lr': core.cosine_lr(row['step'], cfg)})

    method.fit(reference, weights, initial, cfg, batch_provider=lambda step: capture_gim(model, rows(mine, step)),
               step_callback=progress, checkpoint=save, checkpoint_interval=1000, resume=resume, log=log,
               sync_initial=sync_initial, after_backward=after_backward, after_step=after_step)
    if main_proc:
        result = dict(completed=completed, positions=completed*plan['positions_per_step'], resumed_from=start,
                      steps_this_job=completed-start, seconds=time.perf_counter()-started,
                      stop_reason='target_steps' if completed == (SMOKE or STEPS) else 'deadline_or_signal')
        save_json(result, OUT/f'result_{os.environ.get("SLURM_JOB_ID", "local")}.json'); run.summary.update(result); run.finish()
        say(finished=result)
    dist.barrier(); dist.destroy_process_group()
    if completed != (SMOKE or STEPS): sys.exit(3)                            # a chained --resume job continues


if __name__ == '__main__': main()
