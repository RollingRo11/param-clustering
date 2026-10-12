"""Fisher score plus a layer-local sufficiency term, 5,000 steps; otherwise the Fisher K=4 control.

The Lp term picks which pieces are on. The local term makes the rest removable AT THEIR OWN LAYER: for
the global top-q support (q drawn each step from LOCAL_FRACTIONS), every matrix must still produce its
own output, measured after that layer's own nonlinearity (GELU for mlp.c_fc, linear elsewhere). If each
layer's sparse output matches the intact one, the sparse model's activations match everywhere by
induction, which is the condition under which an intact-point sensor is the right thing to train on.
No masked forward pass of the target is used; the term is algebra on the cached intact activations.
LOCAL_METRIC=avg measures the error in M = E[g g^T] (what downstream reads on average, local_metric.py);
raw measures the plain squared error."""
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
from local_metric import metric_factors

K = int(os.environ.get('FISHER_K', '4'))
COEFF = float(os.environ.get('LOCAL_COEFF', '1'))
METRIC = os.environ.get('LOCAL_METRIC', 'avg')
FRACTIONS = tuple(float(v) for v in os.environ.get('LOCAL_FRACTIONS', '0.003,0.01,0.03,0.1').split(','))
METRIC_RANK = int(os.environ.get('LOCAL_RANK', '64'))
METRIC_K = int(os.environ.get('LOCAL_METRIC_K', '8'))
METRIC_ROWS = int(os.environ.get('LOCAL_METRIC_ROWS', '4'))     # held-out sequences for the metric
TARGET_STEPS = int(os.environ.get('LOCAL_STEPS', '5000'))        # lower only for smoke tests
OUT = Path(os.environ.get('LOCAL_OUT', ROOT/f'output/local_{METRIC}_c{COEFF:g}_5000'))   # override only for smoke tests


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
                  local_coeff=COEFF, local_fractions=FRACTIONS, local_metric=METRIC)
    assert cfg.p_end==.5 and cfg.steps==8700 and cfg.local_coeff>0
    # Only the local_* settings may differ from the control's objective.
    assert {k: v for k, v in asdict(cfg).items() if k in control['training_config']}==control['training_config']
    # The local error site is each matrix's own output, after that layer's own nonlinearity.
    specs = [dict(nonlinearity='gelu_tanh' if n.endswith('c_fc') else 'linear', metric=None) for n in MODULES]
    metric_info = None
    if METRIC=='avg':
        # Calibrate the metric on held-out sequences, never on the training stream.
        val, _ = tokens('val')
        rows_list = [torch.from_numpy(np.array(val[600+2*j:602+2*j], dtype=np.int64)).cuda() for j in range((METRIC_ROWS+1)//2)]
        factors, metric_info = metric_factors(model, rows_list, k=METRIC_K, rank=METRIC_RANK)
        for spec, n in zip(specs, MODULES): spec['metric'] = factors[n]
        del rows_list, factors
    index = torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    first = capture_fisher(model,batch(data,0),k=K,seed=123)
    reference = [method.slice_entry(e,index) for e in first]
    del first
    plan = dict(sensor='fisher',score=f'|r.h| * RMS over K={K} sampled next tokens of (g_k.u); plain CE gradients, no GIM rules',
                fisher_k=K,training_config=asdict(cfg),
                local_term=dict(coefficient=COEFF,metric=METRIC,fractions=list(FRACTIONS),metric_rank=METRIC_RANK,
                                metric_fisher_k=METRIC_K,metric_rows=METRIC_ROWS,metric_calibration=metric_info,
                                nonlinearity={n: s['nonlinearity'] for n,s in zip(MODULES,specs)},
                                site='each matrix output after its own nonlinearity; attention Q/K are treated as linear writes',
                                selection='global top-q by the same Fisher score KLKeep ranks with, detached'),
                schedule_horizon_steps=cfg.steps,stop_after=TARGET_STEPS,
                control_plan=str(control_path),control_training_config=control['training_config'],
                control_gpu=control['gpu'],training_hardware_matches_control=control['gpu']==torch.cuda.get_device_name(),
                changed_settings={'score':['gim','fisher'],'local_coeff':[0.,COEFF],'local_metric':['none',METRIC]},
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
    run = wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='local-20260915',
                     name=f'parauto-local-{METRIC}-c{COEFF:g}-5000',job_type='training',dir=str(OUT),config=plan,
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
        checkpoint(dict(**identity,factors=state['pieces'],provenance=dict(kind='Fisher score + layer-local sufficiency training',fisher_k=K,
                        score='|r.h| * RMS_k(g_k.u), K sampled next tokens',local_term=plan['local_term'],
                        training_config=asdict(cfg))),out/'model.pth')
        record = dict(step=step,full_checkpoint=str(out/'checkpoint.pt'),model=str(out/'model.pth'),
                      seconds=time.perf_counter()-before)
        save_json(record,OUT/'latest_checkpoint.json')
        print(json.dumps(dict(checkpoint=record)),flush=True)
        run.log({'update':step,'train/checkpoint_seconds':record['seconds']})
    def log(row):
        if row['matrix_index']==0:
            values = dict(total_loss=row['total_loss'],p=row['p'],lr=core.cosine_lr(row['step'],cfg))
            if 'local_normalized' in row: values['local_normalized'], values['local_fraction'] = row['local_normalized'], row['local_fraction']
            if not np.isfinite(list(values.values())).all(): raise RuntimeError('Nonfinite objective')
            run.log({'update':row['step']+1,**{'train/'+k:v for k,v in values.items()}})
    fitted,_ = method.fit(reference,weights,initial,cfg,batch_provider=provider,
                          step_callback=progress,checkpoint=save,checkpoint_interval=500,stop_after=TARGET_STEPS-start,
                          log=log,resume=resume,local=specs)
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
