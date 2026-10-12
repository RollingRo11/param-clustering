"""Matched 67M GIM + soft activity count; Explorer-only, 5k updates, resumable."""
import argparse
from dataclasses import asdict, replace
import gc
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
import wandb
from common import ROOT, HERE, MODULES, TARGET_SHA, credentials, target, tokens, batch, save_json, checkpoint, sha, cpu_tree
from train_pair import core, method
from gim_attribution import capture_gim, RULES

OUT=ROOT/'output/gim_count67_20260914'


def validate(reference, weights, initial, cfg, provider):
    old_path=ROOT/'code/gaussian67_20260914/par-auto/parauto_method.py'
    spec=importlib.util.spec_from_file_location('legacy_gim_count_control',old_path)
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    plain=replace(cfg,count_coeff=0.)
    old_cfg=legacy.TrainConfig(**{k:v for k,v in asdict(plain).items() if k in legacy.TrainConfig.__dataclass_fields__})
    old,_=legacy.fit(reference,weights,initial,old_cfg,stop_after=2,batch_provider=provider,log=lambda _:None)
    new,_=method.fit(reference,weights,initial,plain,stop_after=2,batch_provider=provider,log=lambda _:None)
    for a,b in zip(old,new):
        torch.testing.assert_close(a.read,b.read,rtol=0,atol=0);torch.testing.assert_close(a.write,b.write,rtol=0,atol=0)
    del old,new
    with torch.no_grad(): s=method.scores_of(reference[0],initial[0]).flatten()[:12].double().clone()
    s.requires_grad_();assert torch.autograd.gradcheck(lambda x:method.soft_activity_count(x,cfg.count_tau), (s,),eps=1e-7,atol=1e-4,rtol=1e-3)
    assert float(method.soft_activity_count(torch.zeros(4,device='cuda'),cfg.count_tau))==0
    for a in [1e-5,1e-3,1.]:
        one=method.soft_activity_count(torch.tensor([a],device='cuda'),cfg.count_tau)
        split=method.soft_activity_count(torch.full((4,),a/4,device='cuda'),cfg.count_tau)
        assert split>=one
    full,_=method.fit(reference,weights,initial,cfg,stop_after=2,batch_provider=provider,log=lambda _:None)
    state={}
    def keep(value): state.update(cpu_tree(value))
    half,_=method.fit(reference,weights,initial,cfg,stop_after=1,batch_provider=provider,checkpoint=keep,log=lambda _:None)
    del half
    resumed,_=method.fit(reference,weights,initial,cfg,resume=state,stop_after=1,batch_provider=provider,log=lambda _:None)
    for a,b in zip(full,resumed):
        torch.testing.assert_close(a.read,b.read,rtol=0,atol=0);torch.testing.assert_close(a.write,b.write,rtol=0,atol=0)
    errors=[core.reconstruction_error(w,p) for w,p in zip(weights,full)]
    assert max(errors)<1e-5
    result=dict(passed=True,zero_coefficient_bitwise_legacy_parity=True,resume_bitwise_parity=True,
                count_gradient_check=True,splitting_budget_never_reduces_count=True,
                max_matrix_error=max(errors),legacy_source_sha256=sha(old_path))
    print(json.dumps(dict(validation=result)),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train']);args=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'),'Use an Explorer SLURM allocation'
    credentials();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0);np.random.seed(0)
    OUT.mkdir(parents=True,exist_ok=True)
    control=json.loads((ROOT/'output/gim_1h/plan.json').read_text())
    cfg=method.TrainConfig(**control['training_config'],count_coeff=1.,count_tau=.001)
    assert cfg.steps==8700 and cfg.batch==32768 and cfg.seed==0 and cfg.p_end==.5
    data,manifest=tokens();model=target();assert sum(p.numel() for p in model.parameters())==66931200
    assert control['target_sha256']==TARGET_SHA and control['data_sha256']==manifest['train']['sha256']
    weights=[model.get_submodule(n).weight for n in MODULES]
    counts=[control['component_counts'][n] for n in MODULES]
    initial=[core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    index=torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    provider=lambda step:capture_gim(model,batch(data,step))
    cache=provider(0);reference=[method.slice_entry(e,index) for e in cache];del cache
    if args.mode=='prepare':
        if (OUT/'prepare.json').exists():raise FileExistsError('Preparation already completed')
        validation=validate(reference,weights,initial,cfg,provider)
        gc.collect();torch.cuda.empty_cache()
        profiles={name:[] for name in ['gim','gim_count']}
        for order in [('gim','gim_count'),('gim_count','gim')]:
            for name in order:
                config=replace(cfg,count_coeff=0. if name=='gim' else 1.)
                ticks=[];torch.cuda.synchronize();tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                def mark(_):
                    nonlocal tick
                    torch.cuda.synchronize();now=time.perf_counter();ticks.append(now-tick);tick=now;return False
                fitted,_=method.fit(reference,weights,initial,config,stop_after=30,batch_provider=provider,step_callback=mark,log=lambda _:None)
                row=dict(seconds_per_step=float(np.mean(ticks[5:])),peak_allocated_bytes=torch.cuda.max_memory_allocated())
                profiles[name].append(row);print(json.dumps(dict(profile=name,**row)),flush=True)
                del fitted;gc.collect();torch.cuda.empty_cache()
        means={n:float(np.mean([v['seconds_per_step'] for v in xs])) for n,xs in profiles.items()}
        save_json(dict(validation=validation,profiles=profiles,mean_seconds=means,relative_to_gim={n:v/means['gim'] for n,v in means.items()},
                       training_config=asdict(cfg),gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID']),OUT/'prepare.json')
        print('GIM_COUNT67_PREPARED',flush=True);return
    prep=json.loads((OUT/'prepare.json').read_text());assert prep['validation']['passed']
    plan=dict(sensor='gim',rules=RULES,training_config=asdict(cfg),steps_to_run=5000,
              initial_normalization='global mean soft count divided by 24; count and summed Lp each start at 24 on fixed reference',
              count_formula='s/(s+count_tau), s=abs(GIM component attribution)',
              count_coefficient=1.,count_tau=.001,reference_index_seed=123,components=dict(zip(MODULES,counts)),
              target_sha256=TARGET_SHA,data_sha256=manifest['train']['sha256'],
              source_manifest_sha256=sha(HERE.parent/'source_manifest.json'),initialization='fresh seed-0 projected factors; no warm start')
    resume=None
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text())==json.loads(json.dumps(plan)),'Immutable plan mismatch'
        if (OUT/'latest_checkpoint.json').exists():
            saved=torch.load(json.loads((OUT/'latest_checkpoint.json').read_text())['full_checkpoint'],map_location='cpu',weights_only=False)
            resume=saved['state'];del saved
    else:save_json(plan,OUT/'plan.json')
    start=resume['next_step'] if resume else 0
    if start>=5000:print('Already complete',flush=True);return
    previous=json.loads((OUT/'wandb.json').read_text()) if (OUT/'wandb.json').exists() else None
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='gim-count67-20260914',name='gim-plus-count-5000',
                   job_type='training',dir=str(OUT),config=plan,id=previous['id'] if previous else None,resume='must' if previous else None)
    save_json(dict(id=run.id,url=run.url),OUT/'wandb.json');run.define_metric('update');run.define_metric('train/*',step_metric='update')
    stopped=False;completed=start
    def request_stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    deadline=float(os.environ.get('COUNT_DEADLINE','inf'))
    torch.cuda.synchronize();began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def mark(step):
        nonlocal completed
        completed=step
        if step%100==0:
            torch.cuda.synchronize();elapsed=time.perf_counter()-began
            row=dict(completed=step,positions=step*32768,seconds=elapsed,seconds_per_step=elapsed/(step-start),peak_allocated_bytes=torch.cuda.max_memory_allocated())
            save_json(row,OUT/'progress.json');print(json.dumps(row),flush=True)
            run.log({'update':step,**{'train/'+k:v for k,v in row.items()}})
        return stopped or time.time()>deadline
    def save(state):
        step=state['next_step'];dest=OUT/'checkpoints'/f'{step:06d}'
        identity=dict(method='parauto',sensor='gim',completed=step,training_positions=step*32768,target_sha256=TARGET_SHA,data_sha256=manifest['train']['sha256'])
        checkpoint(dict(**identity,state=state),dest/'checkpoint.pt')
        checkpoint(dict(**identity,factors=state['pieces'],provenance=dict(kind='GIM plus global soft activity count',rules=RULES,training_config=asdict(cfg))),dest/'model.pth')
        save_json(dict(step=step,full_checkpoint=str(dest/'checkpoint.pt'),model=str(dest/'model.pth')),OUT/'latest_checkpoint.json')
    aggregate={}
    def log(row):
        if row['matrix_index']==0:aggregate.clear()
        for k in ['concentration_normalized','count_normalized','soft_activity_count','batch_threshold_count']:
            aggregate[k]=aggregate.get(k,0.)+row[k]
        if row['matrix_index']==len(MODULES)-1:
            assert np.isfinite(row['total_loss'])
            metrics=dict(total_loss=row['total_loss'],p=row['p'],**aggregate)
            save_json(dict(step=row['step']+1,**metrics),OUT/'loss_progress.json')
            run.log({'update':row['step']+1,**{'train/'+k:v for k,v in metrics.items()}})
    fitted,_=method.fit(reference,weights,initial,cfg,resume=resume,stop_after=5000-start,batch_provider=provider,
                       checkpoint=save,checkpoint_interval=1000,step_callback=mark,log=log)
    errors={n:core.reconstruction_error(w,p) for n,w,p in zip(MODULES,weights,fitted)};assert max(errors.values())<1e-5
    result=dict(completed=completed,complete=completed==5000,positions=completed*32768,seconds=time.perf_counter()-began,
                reconstruction_errors=errors,job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name())
    save_json(result,OUT/'result.json');run.summary.update(result);run.finish()
    print(json.dumps(dict(result=result)),flush=True)
    if completed<5000:raise SystemExit(75)


if __name__=='__main__':main()
