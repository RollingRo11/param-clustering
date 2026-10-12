"""Controlled GIM p_end=.1 experiment; all other settings match the .5 run."""
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

OUT = ROOT/'output/gim_p010_5000'


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
    control_path = ROOT/'output/gim_1h/plan.json'
    control = json.loads(control_path.read_text())
    assert control['sensor']=='gim' and control['rules']==RULES
    assert control['target_sha256']==TARGET_SHA and control['data_sha256']==manifest['train']['sha256']
    assert control['component_counts']==dict(zip(MODULES,counts)) and control['reference_index_seed']==123
    cfg = method.TrainConfig(**control['training_config'])
    assert cfg.p_end==.5 and cfg.steps==8700
    cfg.p_end = .1
    assert {k for k,v in asdict(cfg).items() if v!=control['training_config'][k]}=={'p_end'}
    first = providers['gim'](0)
    reference = [method.slice_entry(e,index) for e in first]
    del first
    plan = dict(sensor='gim',rules=RULES,training_config=asdict(cfg),
                schedule_horizon_steps=cfg.steps,stop_after=5000,
                control_plan=str(control_path),control_training_config=control['training_config'],
                control_gpu=control['gpu'],training_hardware_matches_control=control['gpu']==torch.cuda.get_device_name(),
                changed_settings={'p_end':[.5,.1]},
                schedule_note='Same 8700-step LR/p horizon as control. Change only p_end; both evaluated at 5000.',
                deadline_unix=deadline,checkpoint_interval=1000,target_sha256=TARGET_SHA,
                data_sha256=manifest['train']['sha256'],gpu=torch.cuda.get_device_name(),
                component_counts=dict(zip(MODULES,counts)),reference_index_seed=123,
                initialization='same seed-0 projected factors as control; same fixed reference positions',
                job_id=os.environ.get('SLURM_JOB_ID'))
    save_json(plan,OUT/'plan.json'); print(json.dumps(dict(plan=plan)),flush=True)
    run = wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='gim-sparsity-20260910',
                     name='parauto-gim-p010-5000',job_type='training',dir=str(OUT),config=plan)
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
        checkpoint(dict(**identity,factors=state['pieces'],provenance=dict(kind='controlled stronger sparsity GIM training',
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
                          step_callback=progress,checkpoint=save,checkpoint_interval=1000,stop_after=5000,log=log)
    torch.cuda.synchronize()
    result = dict(completed=completed,positions=completed*32768,seconds=time.perf_counter()-started,
                  stop_reason='signal' if stop else 'time_budget' if time.time()>=deadline else 'target_steps',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),training_config=asdict(cfg),
                  final_lr=core.cosine_lr(completed-1,cfg),data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
    save_json(result,OUT/'result.json'); run.summary.update(result); run.finish()
    print(json.dumps(dict(finished=result)),flush=True)
    if completed!=5000: raise RuntimeError('Incomplete run; do not publish a matched-step evaluation')


if __name__ == '__main__': main()
