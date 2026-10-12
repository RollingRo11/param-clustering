"""Exact single-component ablation ranking, frozen 67M, Explorer only.

One intervention removes one rank-one component at ALL context positions from
its matrix in the intact target. Measure KL separately at all output positions.
Rank once, then delete least-impact first; never recompute ranking after deletion.
Per-token ranking is an output-to-source transfer, separately labeled; context
ranking/removal preserves the exact intervention unit. See individual67.md.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['PAIR_ROOT'])
sys.path.insert(0,str(ROOT/'code/gap67_20260915'))
import gap67 as audit
import numpy as np
import torch
from common import MODULES,TARGET_SHA,target,tokens,save_json,sha,credentials,upload

SPECS={
 'vpd':('vpd',400000,'checkpoints/published_vs_finished/vpd.pth'),
 'vpd5k':('vpd',5000,'checkpoints/step_5000/vpd.pth'),
 'gim1b':('parauto',30518,'checkpoints/published_vs_finished/parauto.pth'),
 'gim5k':('parauto',5000,'output/gim_1h/checkpoints/005000/model.pth'),
}
FIRST,SEQUENCES=440,8
BUDGETS=[0,32,64,128,200,256,512,1024,2048,4096,8192,16384,32768]


@torch.no_grad()
def kl_positions(logits,base_logits):
    # Double precision and centering make tiny individual effects rankable.
    # Keep vocabulary temporaries bounded to 32 positions per chunk.
    result=[]
    for start in range(0,logits.shape[1],32):
        ref=base_logits[:,start:start+32].double()
        lp=ref.log_softmax(-1);prob=lp.exp()
        delta=logits[:,start:start+32].double()-ref
        mu=(prob*delta).sum(-1,keepdim=True)
        value=torch.logsumexp(lp+delta-mu,-1)
        result.append(value.clamp_min(0).float())
    return torch.cat(result,-1)


@torch.no_grad()
def capture(model,ids):
    blocks,hooks={},[]
    try:
        for i,block in enumerate(model.h):
            def hook(_m,args,_out,i=i):blocks[i]=args[0].detach()
            hooks.append(block.register_forward_hook(hook))
        baseline=model(ids)[0]
    finally:
        for hook in hooks:hook.remove()
    return baseline,blocks


@contextmanager
def subtract_component(module,piece,indices,scale=1.):
    def hook(_m,args,output):
        reads=piece.read[:,indices].T[:,None]
        writes=piece.write[:,indices].T[:,None]
        a=(args[0]*reads).sum(-1,keepdim=True)
        return output-scale*a*writes
    handle=module.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


@torch.no_grad()
def ablated_batch(model,blocks,piece,module_index,indices,scale=1.):
    layer=module_index//6
    x=blocks[layer].expand(len(indices),-1,-1)
    with subtract_component(model.get_submodule(MODULES[module_index]),piece,indices,scale):
        for block in model.h[layer:]:x=block(x)
        return model.lm_head(model.ln_f(x))


@torch.no_grad()
def verify(model,ids,baseline,blocks,pieces):
    checks={}
    # Check both early attention and late nonlinear module suffixes against
    # complete model forwards with exactly the same single ablation.
    for index in [0,5,22,23]:
        p=pieces[index];idx=torch.tensor([0,min(17,p.count-1)],device='cuda')
        suffix=ablated_batch(model,blocks,p,index,idx)
        with subtract_component(model.get_submodule(MODULES[index]),p,idx):
            full=model(ids.expand(2,-1))[0]
        err=float((suffix-full).abs().max())
        torch.testing.assert_close(suffix,full,rtol=2e-5,atol=2e-5)
        checks[MODULES[index]+'_suffix_max_error']=err
        untouched=ablated_batch(model,blocks,p,index,idx,scale=0.)
        floor=float(kl_positions(untouched,baseline).max())
        assert floor<1e-8,floor
        checks[MODULES[index]+'_batch_numeric_kl_floor']=floor
    return checks


@torch.no_grad()
def curve(model,ids,pieces,baseline,scores,trainer):
    counts=[p.count for p in pieces];total=sum(counts)
    token_order=scores.T.cuda().argsort(dim=-1,descending=True,stable=True)
    global_order=scores.mean(-1).cuda().argsort(descending=True,stable=True)[None].expand(ids.shape[1],-1)
    result=dict(curves={},native=None)
    if trainer is not None:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            cached=trainer.model(ids,cache_type='input')
            ci=trainer.model.calc_causal_importances(cached.cache,sampling='continuous').lower_leaky
            native=torch.cat([ci[n][0].float() for n in MODULES],-1)>0
        pred=audit.run(model,ids,pieces,native.float())
        result['native']=dict(kl=float(kl_positions(pred,baseline).mean()),count_mean=float(native.sum(-1).float().mean()))
    for label,order in [('context_unit',global_order),('output_to_source_transfer',token_order)]:
        entries=[]
        for kept in sorted(set(BUDGETS+[total]),reverse=True):
            if kept>total:continue
            mask=audit.shared.support(order,kept).float()
            pred=audit.run(model,ids,pieces,mask)
            value=float(kl_positions(pred,baseline).mean())
            # Delta-retained curve: the same original target used by ranking.
            # This is a diagnostic with an uncharged remainder, not primary L0.
            handles=[]
            try:
                for name,p,m in zip(MODULES,pieces,mask.split(counts,-1),strict=True):
                    remainder=(model.get_submodule(name).weight-p.write@p.read.T).detach()
                    def add_remainder(_m,args,out,remainder=remainder):return out+args[0]@remainder.T
                    # audit.run's hook is registered later, so use its context
                    # first, then add the remainder to its replaced output.
                    handles.append((name,add_remainder))
                hooks=[]
                with audit.intervention(model,pieces,audit.masks_of(mask,pieces)):
                    try:
                        hooks=[model.get_submodule(n).register_forward_hook(fn) for n,fn in handles]
                        with_delta=model(ids)[0]
                    finally:
                        for h in hooks:h.remove()
            finally:handles.clear()
            entries.append(dict(kept=kept,removed=total-kept,kl=value,
                                delta_retained_diagnostic_kl=float(kl_positions(with_delta,baseline).mean())))
        result['curves'][label]=entries
    return result


def archive(out):
    receipts=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ['complete.json','r2_receipts.json','progress.json']:
            receipts.append(upload(p,'experiments/individual67_20260915/'+str(p.relative_to(out.parent))))
    save_json(receipts,out/'r2_receipts.json')
    save_json(dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(out/'comparison.json')),out/'complete.json')
    upload(out/'complete.json','experiments/individual67_20260915/'+out.name+'/complete.json')


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('family',choices=list(SPECS));ap.add_argument('--batch',type=int,default=8)
    args=ap.parse_args();assert os.environ.get('SLURM_JOB_ID')
    credentials();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    out=ROOT/'output/individual67_20260915'/args.family;out.mkdir(parents=True,exist_ok=True)
    model=target();data,manifest=tokens('val');audit.shared.SPECS=SPECS
    pieces,trainer,meta=audit.shared.load(model,args.family)
    plan=dict(family=args.family,checkpoint=meta,target_sha256=TARGET_SHA,val_sha256=manifest['val']['sha256'],
              source_sha256=sha(__file__),first=FIRST,sequences=SEQUENCES,intervention_batch=args.batch,
              components=sum(p.count for p in pieces),parameters=sum(p.numel() for p in model.parameters()),
              ranking='single-component context-wide ablation from intact target; fixed ranking then cumulative deletion',
              primary_curve='selected pieces only, no free VPD Delta',
              note='gim1b is original CE-attribution par-auto, 1B positions / 30518 updates; gim5k uses GIM rules')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan
    else:save_json(plan,out/'plan.json')
    deadline=time.time()+float(os.environ.get('INDIVIDUAL_SECONDS','6600'))
    started=time.perf_counter();completed_components=0;records=[]
    for first in range(FIRST,FIRST+SEQUENCES):
        dest=out/f'{first}.json'
        if dest.exists():records.append(json.loads(dest.read_text()));continue
        rows=torch.tensor(np.array(data[first:first+1],dtype=np.int64),device='cuda');ids=rows[:,:-1]
        baseline,blocks=capture(model,ids)
        if not (out/'validation.json').exists():save_json(verify(model,ids,baseline,blocks,pieces),out/'validation.json')
        parts=[]
        for i,(name,p) in enumerate(zip(MODULES,pieces,strict=True)):
            path=out/f'{first}_{i:02d}_effects.npy'
            if path.exists():parts.append(np.load(path));continue
            values=np.empty((p.count,512),dtype=np.float32)
            for c in range(0,p.count,args.batch):
                indices=torch.arange(c,min(c+args.batch,p.count),device='cuda')
                logits=ablated_batch(model,blocks,p,i,indices)
                values[c:c+len(indices)]=kl_positions(logits,baseline).cpu().numpy()
                del logits
                completed_components+=len(indices)
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,values)
            os.replace(temp,path);parts.append(values)
            elapsed=time.perf_counter()-started
            progress=dict(family=args.family,sequence=first,module=name,components_this_invocation=completed_components,
                          elapsed=elapsed,components_per_second=completed_components/max(elapsed,1e-9),job_id=os.environ['SLURM_JOB_ID'])
            save_json(progress,out/'progress.json');print(json.dumps(progress),flush=True)
            if time.time()>deadline:return 75
        effects=np.concatenate(parts,axis=0)
        result=curve(model,ids,pieces,baseline,torch.from_numpy(effects),trainer)
        result.update(sequence=first,effect_files=[dict(path=f'{first}_{i:02d}_effects.npy',sha256=sha(out/f'{first}_{i:02d}_effects.npy')) for i in range(24)],
                      individual_effect_mean=float(effects.mean()),zero_or_tiny_fraction=float((effects<1e-12).mean()))
        save_json(result,dest);records.append(result)
        print(json.dumps(dict(sequence=first,family=args.family,curves=result['curves'])),flush=True)
    save_json(dict(plan=plan,records=records),out/'comparison.json');archive(out)
    print('INDIVIDUAL67_COMPLETE',args.family,flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
