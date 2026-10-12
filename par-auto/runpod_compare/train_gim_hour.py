"""One-hour GIM run, immutable 1,000-step checkpoints, native W&B metrics."""
from dataclasses import asdict
import gc
import json
import math
import os
import random
import signal
import time
import numpy as np
import torch
import wandb
from common import ROOT, MODULES, TARGET_SHA, credentials, tokens, batch, target, checkpoint, save_json
from train_pair import core, method, capture
from gim_attribution import capture_gim, RULES

OUT = ROOT/'output/gim_1h'


def main():
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists(): raise FileExistsError('Refusing to overwrite an existing GIM run')
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
    index = torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    providers = {'gradient':lambda step:capture(model,batch(data,step)),
                 'gim':lambda step:capture_gim(model,batch(data,step))}
    profiles = {}; reference = None
    for sensor in ('gradient','gim'):
        first = providers[sensor](0)
        reference = [method.slice_entry(e,index) for e in first]
        del first
        cfg = method.TrainConfig(steps=1000,batch=32768,precision='bf16',projection_precision='fp32',log_interval=100)
        ticks = []; torch.cuda.synchronize(); tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        def mark(completed):
            nonlocal tick
            torch.cuda.synchronize(); now=time.perf_counter(); ticks.append(now-tick); tick=now
            return stop
        fitted,_ = method.fit(reference,weights,initial,cfg,batch_provider=providers[sensor],
                              stop_after=20,step_callback=mark,log=lambda _:None)
        profiles[sensor] = dict(seconds_per_step=float(np.mean(ticks[5:])),
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),steps=len(ticks))
        del fitted; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(dict(profile=sensor,**profiles[sensor])),flush=True)
    if stop or time.time()+120 > deadline: raise RuntimeError('Insufficient time after profiling')
    # Slightly long horizon ensures we spend the available hour training.
    # Time/signal stop wins; this is a time-budget run, not a full short schedule.
    available = deadline-time.time()
    steps = min(len(data)//64, math.ceil(1.10*available/profiles['gim']['seconds_per_step']/100)*100)
    cfg = method.TrainConfig(steps=steps,batch=32768,precision='bf16',projection_precision='fp32',log_interval=100)
    plan = dict(sensor='gim',rules=RULES,training_config=asdict(cfg),profiles=profiles,
                schedule_horizon_steps=steps,estimated_steps_in_budget=int(available/profiles['gim']['seconds_per_step']),
                schedule_note='Cosine LR 1e-3 to 1e-5 over profiled horizon +10%; p 1 to .5 over first 80%. Stop at time budget.',
                deadline_unix=deadline,checkpoint_interval=1000,target_sha256=TARGET_SHA,
                data_sha256=manifest['train']['sha256'],gpu=torch.cuda.get_device_name(),
                component_counts=dict(zip(MODULES,counts)),reference_index_seed=123,
                initialization='seed-0 projected factors; both 20-step profiles discarded',
                training_data='same globally shuffled stratified file and batch order as previous runs',
                job_id=os.environ.get('SLURM_JOB_ID'))
    save_json(plan,OUT/'plan.json'); print(json.dumps(dict(plan=plan)),flush=True)
    run = wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='gim-1h-20260910',
                     name='parauto-gim-1hour',job_type='training',dir=str(OUT),config=plan)
    save_json(dict(id=run.id,url=run.url),OUT/'wandb.json')
    run.define_metric('update'); run.define_metric('train/*',step_metric='update')
    completed = 0; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    def progress(value):
        nonlocal completed
        completed = value
        if value%100 == 0:
            torch.cuda.synchronize()
            row = dict(update=value,positions=value*32768,seconds=time.perf_counter()-started,
                       seconds_per_step=(time.perf_counter()-started)/value,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),remaining_seconds=max(0.,deadline-time.time()))
            save_json(row,OUT/'progress.json')
            run.log({'update':value,**{'train/'+k:v for k,v in row.items() if k!='update'}})
            print(json.dumps(row),flush=True)
        return stop or time.time() >= deadline
    def save(state):
        step = state['next_step']; before = time.perf_counter()
        out = OUT/'checkpoints'/f'{step:06d}'
        identity = dict(method='parauto',sensor='gim',completed=step,training_positions=step*32768,
                        data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
        checkpoint(dict(**identity,state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'checkpoint.pt')
        checkpoint(dict(**identity,factors=state['pieces'],provenance=dict(kind='one-hour GIM training',
                        rules=RULES,training_config=asdict(cfg))),out/'model.pth')
        record = dict(step=step,full_checkpoint=str(out/'checkpoint.pt'),model=str(out/'model.pth'),
                      seconds=time.perf_counter()-before)
        save_json(record,OUT/'latest_checkpoint.json')
        print(json.dumps(dict(checkpoint=record)),flush=True)
        run.log({'update':step,'train/checkpoint_seconds':record['seconds']})
    def log(row):
        if row['matrix_index']==0:
            values = dict(total_loss=row['total_loss'],p=row['p'],lr=core.cosine_lr(row['step'],cfg))
            if not np.isfinite(list(values.values())).all(): raise RuntimeError('Nonfinite GIM objective')
            run.log({'update':row['step']+1,**{'train/'+k:v for k,v in values.items()}})
    fitted,_ = method.fit(reference,weights,initial,cfg,batch_provider=providers['gim'],
                          step_callback=progress,checkpoint=save,checkpoint_interval=1000,log=log)
    torch.cuda.synchronize()
    result = dict(completed=completed,positions=completed*32768,seconds=time.perf_counter()-started,
                  stop_reason='signal' if stop else 'time_budget' if time.time()>=deadline else 'schedule_end',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),training_config=asdict(cfg),
                  final_lr=core.cosine_lr(completed-1,cfg),data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
    save_json(result,OUT/'result.json'); run.summary.update(result); run.finish()
    print(json.dumps(dict(finished=result)),flush=True)


if __name__ == '__main__': main()
