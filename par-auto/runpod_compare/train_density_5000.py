"""Fisher K=4 score plus VPD's usage-density pricing, 5,000 steps; otherwise the Fisher K=4 control.

Each piece's cost is its mean activity over the batch times (1 + beta * log2(1 + batch total)), the
shape of VPD's ImportanceMinimalityLoss (beta = 0.5 in its published 400k run). Doubling a piece's
average activity more than doubles its price, so heavily used pieces are pushed down and activity
spreads over more, rarer pieces. Scores are divided by a fixed per-matrix scale first, since VPD's
causal importances live in [0, 1] and our attribution magnitudes carry each matrix's own scale.
Matched arm: fisher_k4_5000 (same score, same schedule, beta = 0)."""
from dataclasses import asdict, fields, replace
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
from fisher_attribution import capture_fisher

K = int(os.environ.get('FISHER_K', '4'))
BETA = float(os.environ.get('DENSITY_BETA', '0.5'))          # VPD's published value
TARGET_STEPS = int(os.environ.get('DENSITY_STEPS', '5000'))  # lower only for smoke tests
OUT = Path(os.environ.get('DENSITY_OUT', ROOT/f'output/density_b{BETA:g}_5000'))


def main():
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    # --resume continues from the latest full checkpoint (a second job after a time-budget stop).
    resuming = '--resume' in sys.argv
    if (OUT/'plan.json').exists() != resuming:
        raise FileExistsError('Use --resume exactly when a Fisher run already exists')
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
    provider = lambda step:capture_fisher(model,batch(data,step),k=K,seed=1_000_003+step)
    control_path = ROOT/'output/gim_1h/plan.json'
    control = json.loads(control_path.read_text())
    assert control['sensor']=='gim' and control['rules']==RULES
    assert control['target_sha256']==TARGET_SHA and control['data_sha256']==manifest['train']['sha256']
    assert control['component_counts']==dict(zip(MODULES,counts)) and control['reference_index_seed']==123
    known = {f.name for f in fields(method.TrainConfig)}
    cfg = replace(method.TrainConfig(**{k: v for k, v in control['training_config'].items() if k in known}),
                  density_beta=BETA)
    assert cfg.p_end==.5 and cfg.steps==8700 and cfg.density_beta>0
    # Only density_beta may differ from the control's objective.
    assert {k: v for k, v in asdict(cfg).items() if k in control['training_config']}==control['training_config']
    index = torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    first = capture_fisher(model,batch(data,0),k=K,seed=123)
    reference = [method.slice_entry(e,index) for e in first]
    del first
    plan = dict(sensor='fisher',score=f'|r.h| * RMS over K={K} sampled next tokens of (g_k.u); plain CE gradients, no GIM rules',
                fisher_k=K,training_config=asdict(cfg),
                density_term=dict(beta=BETA,shape='sum_c mean_t a^p * (1 + beta*log2(1 + sum_t a^p))',
                                  source='VPD ImportanceMinimalityLoss beta (published run: beta 0.5, coeff 2e-4, pnorm 2 -> 0.4)',
                                  scale='scores divided by each matrix mean initial score, so the log sees a dimensionless activity',
                                  matched_arm='output/fisher_k4_5000'),
                schedule_horizon_steps=cfg.steps,stop_after=TARGET_STEPS,
                control_plan=str(control_path),control_training_config=control['training_config'],
                control_gpu=control['gpu'],training_hardware_matches_control=control['gpu']==torch.cuda.get_device_name(),
                changed_settings={'score':['gim','fisher'],'density_beta':[0.,BETA]},
                schedule_note='Same 8700-step LR/p horizon, p_end=.5, batch, seed and data as the GIM control; the matched arm is fisher_k4_5000. Evaluated at 5000.',
                deadline_unix=deadline,checkpoint_interval=1000,target_sha256=TARGET_SHA,
                data_sha256=manifest['train']['sha256'],gpu=torch.cuda.get_device_name(),
                component_counts=dict(zip(MODULES,counts)),reference_index_seed=123,
                initialization='same seed-0 projected factors as control; same fixed reference positions',
                job_id=os.environ.get('SLURM_JOB_ID'))
    resume, start = None, 0
    if resuming:
        saved_plan = json.loads((OUT/'plan.json').read_text())
        # JSON round-trip: tuples (joint_fractions) come back as lists.
        assert saved_plan['training_config']==json.loads(json.dumps(plan['training_config'])) and saved_plan['component_counts']==plan['component_counts']
        full = torch.load(json.loads((OUT/'latest_checkpoint.json').read_text())['full_checkpoint'],
                          map_location='cpu', weights_only=False)
        resume, start = full['state'], full['completed']
        torch.set_rng_state(full['torch_rng']); torch.cuda.set_rng_state(full['cuda_rng'])
        del full
        print(json.dumps(dict(resumed_from=start)),flush=True)
    else:
        save_json(plan,OUT/'plan.json'); print(json.dumps(dict(plan=plan)),flush=True)
    previous = json.loads((OUT/'wandb.json').read_text()) if resuming else None
    run = wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='density-20260916',
                     name=f'parauto-density-b{BETA:g}-5000',job_type='training',dir=str(OUT),config=plan,
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
        identity = dict(method='parauto',sensor='fisher',completed=step,training_positions=step*32768,
                        data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
        checkpoint(dict(**identity,state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'checkpoint.pt')
        checkpoint(dict(**identity,factors=state['pieces'],provenance=dict(kind='Fisher score + usage-density pricing',fisher_k=K,
                        score='|r.h| * RMS_k(g_k.u), K sampled next tokens',density_term=plan['density_term'],
                        training_config=asdict(cfg))),out/'model.pth')
        record = dict(step=step,full_checkpoint=str(out/'checkpoint.pt'),model=str(out/'model.pth'),
                      seconds=time.perf_counter()-before)
        save_json(record,OUT/'latest_checkpoint.json')
        print(json.dumps(dict(checkpoint=record)),flush=True)
        run.log({'update':step,'train/checkpoint_seconds':record['seconds']})
    def log(row):
        if row['matrix_index']==0:
            values = dict(total_loss=row['total_loss'],p=row['p'],lr=core.cosine_lr(row['step'],cfg))
            if not np.isfinite(list(values.values())).all(): raise RuntimeError('Nonfinite Fisher objective')
            run.log({'update':row['step']+1,**{'train/'+k:v for k,v in values.items()}})
    fitted,_ = method.fit(reference,weights,initial,cfg,batch_provider=provider,
                          step_callback=progress,checkpoint=save,checkpoint_interval=500,stop_after=TARGET_STEPS-start,
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
    if completed!=TARGET_STEPS: raise RuntimeError('Incomplete run; do not publish a matched-step evaluation')


if __name__ == '__main__': main()
