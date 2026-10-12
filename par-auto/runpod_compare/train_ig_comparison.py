"""Fresh, matched ordinary-gradient and IG5 training within one allocation."""
from dataclasses import asdict
import gc
import json
import os
import random
import signal
import time
import numpy as np
import torch
import wandb
from common import ROOT, MODULES, TARGET_SHA, credentials, tokens, batch, target, checkpoint, save_json
from train_pair import core, method, capture
from integrated_attribution import capture_integrated

OUT=ROOT/'output/ig5_retraining'


def main():
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists(): raise FileExistsError('Use a fresh experiment output directory')
    deadline=time.time()+50*60
    data,manifest=tokens(); model=target()
    weights=[model.get_submodule(n).weight for n in MODULES]
    counts=[dict(q_proj=1024,k_proj=1024,v_proj=1024,o_proj=1024,c_fc=3072,down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    initial=[core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    index=torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    providers={'gradient':lambda step:capture(model,batch(data,step)),
               'ig5':lambda step:capture_integrated(model,batch(data,step),k=5)}
    references={}; profiles={}
    stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGUSR1,request_stop)
    for sensor in ('gradient','ig5'):
        first=providers[sensor](0)
        references[sensor]=[method.slice_entry(e,index) for e in first]
        del first
        cfg=method.TrainConfig(steps=1000,batch=32768,precision='bf16',projection_precision='fp32',log_interval=100)
        ticks=[]; tick=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        def mark(completed):
            nonlocal tick
            torch.cuda.synchronize(); now=time.perf_counter(); ticks.append(now-tick); tick=now
            return stop
        fitted,_=method.fit(references[sensor],weights,initial,cfg,batch_provider=providers[sensor],
                            stop_after=10,step_callback=mark,log=lambda _:None)
        profiles[sensor]=dict(seconds_per_step=float(np.mean(ticks[3:])),
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),steps=len(ticks))
        del fitted; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(dict(profile=sensor,**profiles[sensor])),flush=True)
    if stop: raise RuntimeError('Stopped during profile')
    # Same complete cosine LR and p schedules for both arms; pilots are discarded.
    available=deadline-time.time()-180
    steps=min(2000,int(available/(1.35*sum(p['seconds_per_step'] for p in profiles.values())))//100*100)
    if steps < 100: raise RuntimeError('Insufficient allocation time for a matched comparison')
    cfg=method.TrainConfig(steps=steps,batch=32768,precision='bf16',projection_precision='fp32',log_interval=100)
    plan=dict(steps=steps,positions_per_arm=steps*32768,training_config=asdict(cfg),profiles=profiles,
              target_sha256=TARGET_SHA,data_sha256=manifest['train']['sha256'],gpu=torch.cuda.get_device_name(),
              initialization='same seed-0 projected factors; pilots discarded',reference_index_seed=123,
              component_counts=dict(zip(MODULES,counts)),ig_k=5,
              ig_path='all 24 matrix gates jointly 0 -> 1; Gauss-Legendre quadrature',
              score='absolute value of signed integrated scalar-CE attribution',
              scoring_backward='IG score matmuls recomputed with torch checkpoint; original objective preserved',
              deadline=deadline)
    save_json(plan,OUT/'plan.json'); print(json.dumps(dict(plan=plan)),flush=True)
    for sensor in ('ig5','gradient'):
        out=OUT/sensor; out.mkdir(exist_ok=True)
        run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',
                       group='ig5-retraining-20260910',name=f'parauto-{sensor}-fresh-{steps}',
                       job_type='training',dir=str(out),config=dict(sensor=sensor,**plan))
        save_json(dict(id=run.id,url=run.url),out/'wandb.json')
        run.define_metric('update'); run.define_metric('train/*',step_metric='update')
        completed=0; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        def progress(value):
            nonlocal completed
            completed=value
            if value%20==0:
                torch.cuda.synchronize()
                row=dict(update=value,seconds_per_step=(time.perf_counter()-started)/value,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated())
                run.log({'update':value,**{'train/'+k:v for k,v in row.items() if k!='update'}})
                print(json.dumps(dict(sensor=sensor,**row)),flush=True)
            return stop or time.time() > deadline
        def save(state):
            checkpoint(dict(method='parauto',sensor=sensor,completed=state['next_step'],state=state,
                        data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/'checkpoint.pt')
        def log(row):
            if row['matrix_index']==0:
                run.log({'update':row['step']+1,'train/total_loss':row['total_loss'],'train/p':row['p']})
        fitted,_=method.fit(references[sensor],weights,initial,cfg,batch_provider=providers[sensor],
                            step_callback=progress,checkpoint=save,checkpoint_interval=250,log=log)
        torch.cuda.synchronize()
        result=dict(sensor=sensor,completed=completed,positions=completed*32768,complete=completed==steps,
                    seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    training_config=asdict(cfg),data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
        checkpoint(dict(method='parauto',completed=completed,training_positions=completed*32768,
                        factors=[dict(read=p.read,write=p.write) for p in fitted],
                        data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA,
                        provenance=dict(kind='fresh matched sensor-training comparison',sensor=sensor,
                                        training_config=asdict(cfg))),out/'model.pth')
        save_json(result,out/'result.json'); run.summary.update(result); run.finish()
        del fitted; gc.collect(); torch.cuda.empty_cache()
        if completed!=steps: raise RuntimeError('Training interrupted; matched comparison not completed')
    save_json(dict(complete=True,steps=steps,positions_per_arm=steps*32768),OUT/'training_complete.json')


if __name__=='__main__': main()
