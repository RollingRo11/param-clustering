"""Per-piece finite local ablation score in place of GIM, on 1,024 sampled tokens per step.

Each step still reads the same 64 sequences; 16 random positions per sequence enter the loss.
Everything else matches the GIM p=.5 run. Capture is forward-only."""
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import numpy as np
import torch
import wandb
from common import ROOT, MODULES, TARGET_SHA, credentials, tokens, batch, target, checkpoint, save_json
from train_pair import core, method, capture
from gim_attribution import RULES
from finite_ablation import capture_finite

OUT = Path(os.environ.get('FINITE_OUT', ROOT/'output/finite_5000'))   # override only for smoke tests
TOKENS_PER_SEQUENCE = 16


def main():
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    # --resume continues from the latest full checkpoint (a second job after a time-budget stop).
    resuming = '--resume' in sys.argv
    if (OUT/'plan.json').exists() != resuming:
        raise FileExistsError('Use --resume exactly when a finite-ablation run already exists')
    deadline = float(os.environ['GIM_DEADLINE'])
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGUSR1, request_stop)
    data, manifest = tokens(); model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    counts = [dict(q_proj=1024,k_proj=1024,v_proj=1024,o_proj=1024,c_fc=3072,down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    initial = [core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    sampler = lambda seed: torch.Generator(device='cuda').manual_seed(seed)
    provider = lambda step:capture_finite(model,batch(data,step),TOKENS_PER_SEQUENCE,sampler(1_000_003+step))
    control_path = ROOT/'output/gim_1h/plan.json'
    control = json.loads(control_path.read_text())
    assert control['sensor']=='gim' and control['rules']==RULES
    assert control['target_sha256']==TARGET_SHA and control['data_sha256']==manifest['train']['sha256']
    assert control['component_counts']==dict(zip(MODULES,counts)) and control['reference_index_seed']==123
    cfg = method.TrainConfig(**control['training_config'])
    assert cfg.p_end==.5 and cfg.steps==8700
    cfg.batch = 64*TOKENS_PER_SEQUENCE
    assert {k for k,v in asdict(cfg).items() if v!=control['training_config'][k]}=={'batch'}
    # Normalization uses the first 256 sampled tokens of batch 0 (fit slices them).
    reference = capture_finite(model,batch(data,0),TOKENS_PER_SEQUENCE,sampler(123))
    plan = dict(sensor='finite_ablation',score='per-piece finite local ablation (finite_ablation.py)',training_config=asdict(cfg),
                tokens_per_sequence=TOKENS_PER_SEQUENCE,loss_tokens_per_step=cfg.batch,
                schedule_horizon_steps=cfg.steps,stop_after=5000,
                control_plan=str(control_path),control_training_config=control['training_config'],
                control_gpu=control['gpu'],training_hardware_matches_control=control['gpu']==torch.cuda.get_device_name(),
                changed_settings={'score':['gim','finite_ablation'],'batch':[32768,cfg.batch]},
                schedule_note='Same 8700-step LR/p horizon and p_end=.5 as control. Score changes; loss uses 1,024 of the 32,768 tokens read per step.',
                deadline_unix=deadline,checkpoint_interval=1000,target_sha256=TARGET_SHA,
                data_sha256=manifest['train']['sha256'],gpu=torch.cuda.get_device_name(),
                component_counts=dict(zip(MODULES,counts)),reference_index_seed=123,
                initialization='same seed-0 projected factors as control; same fixed reference positions',
                job_id=os.environ.get('SLURM_JOB_ID'))
    resume, start = None, 0
    if resuming:
        saved_plan = json.loads((OUT/'plan.json').read_text())
        assert saved_plan['training_config']==plan['training_config'] and saved_plan['component_counts']==plan['component_counts']
        full = torch.load(json.loads((OUT/'latest_checkpoint.json').read_text())['full_checkpoint'],
                          map_location='cpu', weights_only=False)
        resume, start = full['state'], full['completed']
        torch.set_rng_state(full['torch_rng']); torch.cuda.set_rng_state(full['cuda_rng'])
        del full
        print(json.dumps(dict(resumed_from=start)),flush=True)
    else:
        save_json(plan,OUT/'plan.json'); print(json.dumps(dict(plan=plan)),flush=True)
    previous = json.loads((OUT/'wandb.json').read_text()) if resuming else None
    run = wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='finite-ablation-20260910',
                     name='parauto-finite-5000',job_type='training',dir=str(OUT),config=plan,
                     id=previous['id'] if previous else None,resume='must' if previous else None)
    save_json(dict(id=run.id,url=run.url),OUT/'wandb.json')
    run.define_metric('update'); run.define_metric('train/*',step_metric='update')
    completed = start; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    def progress(value):
        nonlocal completed
        completed = value
        if value%100 == 0:
            torch.cuda.synchronize()
            row = dict(update=value,positions=value*32768,seconds=time.perf_counter()-started,
                       seconds_per_step=(time.perf_counter()-started)/max(1,value-start),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),remaining_seconds=max(0.,deadline-time.time()))
            save_json(row,OUT/'progress.json')
            run.log({'update':value,**{'train/'+k:v for k,v in row.items() if k!='update'}})
            print(json.dumps(row),flush=True)
        return stop or time.time() >= deadline
    def save(state):
        step = state['next_step']; before = time.perf_counter()
        out = OUT/'checkpoints'/f'{step:06d}'
        identity = dict(method='parauto',sensor='finite_ablation',completed=step,training_positions=step*32768,
                        data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
        checkpoint(dict(**identity,state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'checkpoint.pt')
        checkpoint(dict(**identity,factors=state['pieces'],provenance=dict(kind='finite local ablation score training',
                        score='per-piece finite local ablation',training_config=asdict(cfg))),out/'model.pth')
        record = dict(step=step,full_checkpoint=str(out/'checkpoint.pt'),model=str(out/'model.pth'),
                      seconds=time.perf_counter()-before)
        save_json(record,OUT/'latest_checkpoint.json')
        print(json.dumps(dict(checkpoint=record)),flush=True)
        run.log({'update':step,'train/checkpoint_seconds':record['seconds']})
    def log(row):
        if row['matrix_index']==0:
            values = dict(total_loss=row['total_loss'],p=row['p'],lr=core.cosine_lr(row['step'],cfg))
            if not np.isfinite(list(values.values())).all(): raise RuntimeError('Nonfinite finite-ablation objective')
            run.log({'update':row['step']+1,**{'train/'+k:v for k,v in values.items()}})
    fitted,_ = method.fit(reference,weights,initial,cfg,batch_provider=provider,
                          step_callback=progress,checkpoint=save,checkpoint_interval=500,stop_after=5000-start,
                          log=log,resume=resume)
    torch.cuda.synchronize()
    result = dict(completed=completed,positions=completed*32768,seconds=time.perf_counter()-started,
                  resumed_from=start,steps_this_job=completed-start,
                  stop_reason='signal' if stop else 'time_budget' if time.time()>=deadline else 'target_steps',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),training_config=asdict(cfg),
                  final_lr=core.cosine_lr(completed-1,cfg),data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
    save_json(result,OUT/f'result_{os.environ.get("SLURM_JOB_ID","local")}.json')
    save_json(result,OUT/'result.json'); run.summary.update(result); run.finish()
    print(json.dumps(dict(finished=result)),flush=True)
    if completed!=5000: raise RuntimeError('Incomplete run; do not publish a matched-step evaluation')


if __name__ == '__main__': main()
