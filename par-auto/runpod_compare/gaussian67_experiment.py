"""Explorer-only 67M validation, matched whole-update profiling, resumable 5k trials."""
from dataclasses import asdict
import argparse
import gc
import json
import os
import signal
import time
import numpy as np
import torch
import wandb
from common import ROOT, HERE, MODULES, TARGET_SHA, tokens, batch, target, save_json, checkpoint, sha, credentials
from train_pair import core, method
from gim_attribution import capture_gim
from gaussian_attribution import capture_gaussian, calibrate_boundaries


EXPERIMENT = ROOT/'output/gaussian67_20260914'
ARMS = {'gaussian': 0., 'boundary001': .01}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode', choices=['prepare','train'])
    ap.add_argument('--arm', choices=list(ARMS)); ap.add_argument('--stop-after', type=int, default=5000)
    args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run inside an Explorer SLURM allocation')
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(0); np.random.seed(0)
    data, manifest=tokens(); model=target()
    weights=[model.get_submodule(n).weight for n in MODULES]
    counts=[dict(q_proj=1024,k_proj=1024,v_proj=1024,o_proj=1024,c_fc=3072,down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    initial=[core.initialize_pieces(w,c,0) for w,c in zip(weights,counts)]
    control=json.loads((ROOT/'output/gim_1h/plan.json').read_text())
    cfg=method.TrainConfig(**control['training_config'])
    assert cfg.steps==8700 and cfg.p_end==.5 and cfg.batch==32768 and cfg.lr==1e-3
    assert control['target_sha256']==TARGET_SHA and control['data_sha256']==manifest['train']['sha256']
    index=torch.randperm(32768, device='cuda', generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    EXPERIMENT.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':
        if (EXPERIMENT/'prepare.json').exists(): raise FileExistsError('Preparation already completed')
        from gaussian67_validate import validate
        scales=validate(model,batch(data,0),initial,EXPERIMENT/'validation.json')
        save_json(scales,EXPERIMENT/'boundary_scales.json')
    else:
        if not args.arm: raise ValueError('--arm required')
        prep=json.loads((EXPERIMENT/'prepare.json').read_text())
        if not prep['validation_passed']: raise RuntimeError('Validation failed')
        scales=json.loads((EXPERIMENT/'boundary_scales.json').read_text())
    def provider(sensor, step):
        rows=batch(data,step)
        if sensor=='gim': return capture_gim(model,rows)
        return capture_gaussian(model,rows,seed=1_000_003+step,coefficient=ARMS[sensor],scales=scales)
    def reference(sensor):
        cache=provider(sensor,0)
        ref=[method.slice_entry(e,index) for e in cache]
        # Advanced indexing materializes the selected tensors, releasing the full cache.
        return ref
    if args.mode=='prepare':
        profiles={k:[] for k in ['gim',*ARMS]}
        for order in [('gim','gaussian','boundary001'),('boundary001','gaussian','gim')]:
            for sensor in order:
                ref=reference(sensor); ticks=[]; torch.cuda.synchronize(); tick=time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                def mark(_):
                    nonlocal tick
                    torch.cuda.synchronize(); now=time.perf_counter(); ticks.append(now-tick); tick=now
                    return False
                fitted,_=method.fit(ref,weights,initial,cfg,batch_provider=lambda s:provider(sensor,s),
                    stop_after=30,step_callback=mark,log=lambda _:None)
                row=dict(warm_seconds_per_step=float(np.mean(ticks[5:])),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),steps=30)
                profiles[sensor].append(row); print(json.dumps(dict(profile=sensor,**row)),flush=True)
                del fitted,ref; gc.collect(); torch.cuda.empty_cache()
        means={k:float(np.mean([r['warm_seconds_per_step'] for r in v])) for k,v in profiles.items()}
        ratios={k:v/means['gim'] for k,v in means.items()}
        save_json(dict(validation_passed=True,profiles=profiles,mean_seconds=means,relative_to_gim=ratios,
            training_config=asdict(cfg),target_sha256=TARGET_SHA,data_sha256=manifest['train']['sha256'],
            gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],
            note='Full cache+factor update+projection timings, fresh initial factors each profile; no speed claim until inspected.'),EXPERIMENT/'prepare.json')
        print('GAUSSIAN67_PREPARED',flush=True); return
    out=EXPERIMENT/args.arm; out.mkdir(exist_ok=True)
    plan=dict(sensor='gaussian_fisher',boundary_coefficient=ARMS[args.arm],boundary_scales=scales,
              training_config=asdict(cfg),target_sha256=TARGET_SHA,data_sha256=manifest['train']['sha256'],
              components=dict(zip(MODULES,counts)),stop_after=5000,probe_seed='1000003+step, plus microbatch offset',
              reference_index_seed=123,reference_probe_seed=1000003,
              smoothing='existing (abs(a)+1e-6)^p - 1e-6^p; unsmoothed Gaussian identity approximate near epsilon',
              normalization='same fixed 256 training positions; common kappa_p cancels numerator/denominator',
              gpu=torch.cuda.get_device_name(),source_manifest_sha256=sha(HERE.parent/'source_manifest.json'))
    resume=None; latest=out/'latest_checkpoint.json'
    if (out/'plan.json').exists():
        old=json.loads((out/'plan.json').read_text())
        assert all(old[k]==plan[k] for k in ['training_config','target_sha256','data_sha256','boundary_coefficient','boundary_scales','source_manifest_sha256'])
        if latest.exists():
            saved=torch.load(json.loads(latest.read_text())['full_checkpoint'],map_location='cpu',weights_only=False)
            resume=saved['state']; del saved
    else: save_json(plan,out/'plan.json')
    start=resume['next_step'] if resume else 0
    if start>=args.stop_after:
        print(f'Already at {start} steps',flush=True); return
    previous=json.loads((out/'wandb.json').read_text()) if (out/'wandb.json').exists() else None
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group='gaussian67-20260914',
        name=f'gaussian67-{args.arm}-5000',job_type='training',dir=str(out),config=plan,
        id=previous['id'] if previous else None,resume='must' if previous else None)
    save_json(dict(id=run.id,url=run.url),out/'wandb.json')
    run.define_metric('update'); run.define_metric('train/*',step_metric='update')
    stopped=False
    def request_stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGUSR1,request_stop); signal.signal(signal.SIGTERM,request_stop)
    deadline=float(os.environ.get('GAUSSIAN_DEADLINE','inf'))
    ref=reference(args.arm); torch.cuda.synchronize(); began=time.perf_counter(); completed=start
    torch.cuda.reset_peak_memory_stats()
    def mark(step):
        nonlocal completed
        completed=step
        if step%100==0:
            torch.cuda.synchronize()
            row=dict(completed=step,positions=step*32768,seconds=time.perf_counter()-began,
                     invocation_start=start,seconds_per_step=(time.perf_counter()-began)/(step-start),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated())
            save_json(row,out/'progress.json'); print(json.dumps(row),flush=True)
            run.log({'update':step,**{'train/'+k:v for k,v in row.items()}})
        return stopped or time.time()>deadline
    def save(state):
        step=state['next_step']; dest=out/'checkpoints'/f'{step:06d}'
        identity=dict(method='parauto',sensor='gaussian_fisher',completed=step,training_positions=step*32768,
                      data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA)
        checkpoint(dict(**identity,state=state),dest/'checkpoint.pt')
        checkpoint(dict(**identity,factors=state['pieces'],provenance=dict(kind='Gaussian Fisher training',
                        boundary_coefficient=ARMS[args.arm],training_config=asdict(cfg))),dest/'model.pth')
        save_json(dict(step=step,full_checkpoint=str(dest/'checkpoint.pt'),model=str(dest/'model.pth')),latest)
    def log(row):
        if row['matrix_index']==0:
            if not np.isfinite(row['total_loss']): raise RuntimeError('Nonfinite objective')
            run.log({'update':row['step']+1,'train/loss':row['total_loss'],'train/p':row['p']})
    fitted,_=method.fit(ref,weights,initial,cfg,resume=resume,batch_provider=lambda s:provider(args.arm,s),
        stop_after=args.stop_after-start,step_callback=mark,checkpoint=save,checkpoint_interval=1000,log=log)
    errors={n:core.reconstruction_error(w,p) for n,w,p in zip(MODULES,weights,fitted)}
    result=dict(completed=completed,positions=completed*32768,seconds=time.perf_counter()-began,
        invocation_start=start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        reconstruction_errors=errors,complete=completed>=args.stop_after,
        job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name())
    save_json(result,out/f'result_{start}_{completed}.json'); save_json(result,out/'result.json')
    run.summary.update(result); run.finish(); print(json.dumps(dict(result=result)),flush=True)
    if completed<args.stop_after: raise SystemExit(75)


if __name__=='__main__': main()
