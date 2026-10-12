"""Four matched 5k training runs, selected by actual KLKeep, Explorer only."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import statistics as st
import sys
import time

ROOT=Path(os.environ['PAIR_ROOT'])
sys.path.insert(0,str(ROOT/'code/first_nonlinearity67_20260915'))
import first_nonlinearity67 as first
obj,core=first.obj,first.core
from train_pair import method
from common import MODULES,TARGET_SHA,target,tokens,batch,checkpoint,save_json,sha,cpu_tree,credentials,upload
from gim_attribution import gim_context,ScaleGradient,capture_gim
import numpy as np
import torch
import torch.nn.functional as F
import wandb

EXPERIMENT='objective_train67_20260915_v3'
VARIANTS=['finite_gim','finite_fisher','write','pair_gim']
SAMPLES=64  # per context and matrix: 4096 independent component-position draws/update
MICRO=4


def capture_batch(model,rows,sensor,seed):
    pre,post,norms,prob,handles={},{},{},[],[]
    names=[f'h.{i}.rms_{j}' for i in range(4) for j in [1,2]]+['ln_f']
    isgim=sensor=='gim'
    with gim_context(model,temperature=2. if isgim else 1.,freeze_rms=isgim):
        oldsoft=F.softmax
        def soft(x,*a,**kw):
            y=oldsoft(x,*a,**kw)
            if x.ndim==4:prob.append((x,y))
            return y
        F.softmax=soft
        def mh(n):
            def hook(m,args,out):
                pre[n],post[n]=args[0],out
                scale={'q_proj':.25,'k_proj':.25,'v_proj':.5}.get(n.split('.')[-1])
                if isgim and scale is not None:return ScaleGradient.apply(out,scale)
            return hook
        def nh(n):
            def hook(m,args,out):norms[n]=(args[0],out)
            return hook
        try:
            for n in MODULES:handles.append(model.get_submodule(n).register_forward_hook(mh(n)))
            for n in names:handles.append(model.get_submodule(n).register_forward_hook(nh(n)))
            handles.append(model.wte.register_forward_hook(lambda m,i,o:o.detach().requires_grad_()))
            logits=model(rows[:,:-1])[0]
            labels=rows[:,1:]
            if sensor=='fisher':
                gen=torch.Generator(device='cuda').manual_seed(seed)
                labels=torch.multinomial(logits.detach().flatten(0,1).softmax(-1),1,generator=gen).reshape_as(labels)
            loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction='sum')
            assert len(prob)==4
            grads=torch.autograd.grad(loss,[post[n] for n in MODULES]+[norms[n][1] for n in names]+[p for _,p in prob])
        finally:
            F.softmax=oldsoft
            for h in handles:h.remove()
    result=[]
    with torch.no_grad():
        for j in range(len(rows)):
            ctx=dict(sensor='gim' if isgim else 'true',rows=rows[j:j+1],
                inputs={n:x[j].detach() for n,x in pre.items()},outputs={n:x[j].detach() for n,x in post.items()},
                g={n:g[j].detach() for n,g in zip(MODULES,grads[:24])},
                norms={n:(norms[n][0][j].detach(),norms[n][1][j].detach(),g[j].detach()) for n,g in zip(names,grads[24:33])},att={})
            for layer,((l,p),g) in enumerate(zip(prob,grads[33:])):
                att=model.h[layer].attn;T=512
                q=ctx['outputs'][MODULES[layer*6]].reshape(T,6,128).transpose(0,1)[None]
                k=ctx['outputs'][MODULES[layer*6+1]].reshape(T,6,128).transpose(0,1)[None]
                q,k=att.apply_rotary_pos_emb(q,k,att.rotary_cos[:T][None],att.rotary_sin[:T][None])
                l,p,b=l[j].detach(),p[j].detach(),g[j].detach()
                w=(l.double()-l.double().amax(-1,keepdim=True)).exp()
                def excl(x):return F.pad(x.cumsum(-1)[...,:-1],(1,0))+F.pad(x.flip(-1).cumsum(-1).flip(-1)[...,1:],(0,1))
                rest=excl(w)
                ctx['att'][layer]=dict(q=q[0],k=k[0],logits=l,p=p,b=b,
                    other=(excl(w*b.double())/rest.clamp_min(1e-300)).float(),
                    logrest=(rest.log()+l.double().amax(-1,keepdim=True)).float())
            result.append(ctx)
    return result


def caches(model,rows,variant,step):
    sensor='fisher' if variant=='finite_fisher' else 'gim'
    return [c for i in range(0,len(rows),MICRO) for c in capture_batch(model,rows[i:i+MICRO],sensor,8100000+step*64+i)]


def rotated(model,layer,w,pos):
    att=model.h[layer].attn;u=w.reshape(-1,6,128)
    return u*att.rotary_cos[pos][:,None]+att.rotate_every_two(u)*att.rotary_sin[pos][:,None]


def score(model,ctx,i,p,pos,comp,variant):
    n=MODULES[i];kind=i%6;layer=i//6
    r,w=p.read[:,comp].T,p.write[:,comp].T
    a=(ctx['inputs'][n][pos]*r).sum(-1)
    if variant=='write':return a*w.norm(dim=-1)
    linear=-a*(ctx['g'][n][pos]*w).sum(-1)
    if variant=='pair_gim' or kind==2:return linear
    z=a[:,None]*w
    if kind==4:
        x=ctx['outputs'][n][pos]
        b=ctx['g'][MODULES[i+1]][pos]@model.get_submodule(MODULES[i+1]).weight
        return ((F.gelu(x-z,approximate='tanh')-F.gelu(x,approximate='tanh'))*b).sum(-1)
    if kind in [3,5]:
        nn=f'h.{layer}.rms_2' if kind==3 else (f'h.{layer+1}.rms_1' if layer<3 else 'ln_f')
        rr,_,bb=ctx['norms'][nn];rr,bb=rr[pos],bb[pos];norm=model.get_submodule(nn)
        inv=(rr.square().mean(-1,keepdim=True)+norm.variance_epsilon).rsqrt()
        jz=norm.weight*inv*z
        if ctx['sensor']=='true':jz=jz-norm.weight*inv**3*rr*(rr*z).mean(-1,keepdim=True)
        return linear+((norm(rr-z)-norm(rr)+jz)*bb).sum(-1)
    att=ctx['att'][layer];u=rotated(model,layer,w,pos)
    scale=.25 if ctx['sensor']=='gim' else 1.
    if kind==0:
        d=-a[:,None,None]*torch.einsum('mhd,hsd->mhs',u,att['k'])/math.sqrt(128)
        logits=att['logits'][:,pos,:].permute(1,0,2)
        old=att['p'][:,pos,:].permute(1,0,2);b=att['b'][:,pos,:].permute(1,0,2)
        return scale*(((logits+d).softmax(-1)-old)*b).sum((-1,-2))
    d=-a[:,None,None]*torch.einsum('htd,mhd->mht',att['q'],u)/math.sqrt(128)
    l=att['logits'][:,:,pos].permute(2,0,1);rest=att['logrest'][:,:,pos].permute(2,0,1)
    valid=torch.isfinite(l)&torch.isfinite(rest);odds=torch.where(valid,l-rest,torch.zeros_like(l))
    b=(att['b']-att['other'])[:,:,pos].permute(2,0,1)
    return scale*(((odds+d).sigmoid()-odds.sigmoid())*b*valid).sum((-1,-2))


def pair_score(model,ctx,layer,q,k,pos,comp,keypos,keycomp):
    qi,ki=layer*6,layer*6+1
    aq=(ctx['inputs'][MODULES[qi]][pos]*q.read[:,comp].T).sum(-1)
    ak=(ctx['inputs'][MODULES[ki]][keypos]*k.read[:,keycomp].T).sum(-1)
    uq=rotated(model,layer,q.write[:,comp].T,pos);uk=rotated(model,layer,k.write[:,keycomp].T,keypos)
    att=ctx['att'][layer]
    # GIM's softmax VJP uses temperature two without an extra inverse-temperature.
    prob=(att['logits']/2).softmax(-1)
    gs=prob*(att['b']-(prob*att['b']).sum(-1,keepdim=True))
    interaction=aq[:,None]*ak[:,None]*(uq*uk).sum(-1)/math.sqrt(128)
    return .25*(interaction*gs[:,pos,keypos].T).sum(-1)


def draws(count,gen,n=SAMPLES):
    return torch.randint(512,(n,),device='cuda',generator=gen),torch.randint(count,(n,),device='cuda',generator=gen)


def penalty(x,p):return (x.abs()+1e-6).pow(p)-1e-6**p


def reference(model,ctxs,pieces,variant):
    gen=torch.Generator(device='cuda').manual_seed(19237);refs=[]
    with torch.no_grad():
        for i,p in enumerate(pieces):
            if variant=='pair_gim' and i%6==1:refs.append(refs[-1]);continue
            values=[];weights=[]
            for ctx in ctxs[:4]:
                pos,c=draws(p.count,gen,1024)
                if variant=='pair_gim' and i%6==0:
                    kp=(torch.rand(len(pos),device='cuda',generator=gen)*(pos+1)).long()
                    kc=torch.randint(pieces[i+1].count,pos.shape,device='cuda',generator=gen)
                    s=pair_score(model,ctx,i//6,p,pieces[i+1],pos,c,kp,kc);w=(pos+1).float()
                else:s=score(model,ctx,i,p,pos,c,variant);w=torch.ones_like(s)
                values.append(s);weights.append(w)
            refs.append(dict(scores=torch.cat(values),weights=torch.cat(weights)))
    return refs


def update(model,ctxs,pieces,opt,refs,cfg,variant,step,*,sampling_seed=None,apply_step=True):
    opt.zero_grad(set_to_none=True)
    p=1+(cfg.p_end-1)*min(1.,step/(cfg.steps*.8))
    for group in opt.param_groups:group['lr']=core.cosine_lr(step,cfg)
    gen=torch.Generator(device='cuda').manual_seed(9100000+step if sampling_seed is None else sampling_seed)
    losses=[]
    for i,item in enumerate(pieces):
        if variant=='pair_gim' and i%6==1:continue
        f=method.project(model.get_submodule(MODULES[i]).weight,item,cfg)
        partner=method.project(model.get_submodule(MODULES[i+1]).weight,pieces[i+1],cfg) if variant=='pair_gim' and i%6==0 else None
        terms=[]
        for ctx in ctxs:
            pos,c=draws(item.count,gen)
            if partner is not None:
                kp=(torch.rand(len(pos),device='cuda',generator=gen)*(pos+1)).long()
                kc=torch.randint(partner.count,pos.shape,device='cuda',generator=gen)
                s=pair_score(model,ctx,i//6,f,partner,pos,c,kp,kc);weights=(pos+1).float()
            else:s=score(model,ctx,i,f,pos,c,variant);weights=1.
            terms.append((penalty(s,p)*weights).mean())
        denominator=(penalty(refs[i]['scores'],p)*refs[i]['weights']).mean().clamp_min(1e-10)
        loss=torch.stack(terms).mean()/denominator*(2 if partner is not None else 1)
        assert torch.isfinite(loss)
        loss.backward();losses.append(float(loss.detach()))
        del f,partner,terms,loss
    assert all(torch.isfinite(t.grad).all() for x in pieces for t in [x.read,x.write])
    if apply_step:
        opt.step()
        for n,item in zip(MODULES,pieces):method.project_(model.get_submodule(n).weight,item,cfg)
    return dict(loss=sum(losses),p=p,lr=core.cosine_lr(step,cfg))


def state(pieces,opt,refs,step):
    return dict(pieces=[dict(read=p.read,write=p.write) for p in pieces],optimizer=opt.state_dict(),references=refs,next_step=step)


def create(initial,saved=None):
    pieces=obj.clone(initial,grad=True);opt=torch.optim.Adam([t for p in pieces for t in [p.read,p.write]],lr=.001,betas=(.9,.99))
    if saved:
        with torch.no_grad():
            for p,s in zip(pieces,saved['pieces']):p.read.copy_(s['read']);p.write.copy_(s['write'])
        opt.load_state_dict(saved['optimizer'])
    return pieces,opt


def prepare(model,training,initial,cfg,variant,out):
    rows=batch(training,0);ctxs=caches(model,rows,variant,0)
    # Sample evaluator must reproduce the independently checked dense implementation.
    gen=torch.Generator(device='cuda').manual_seed(731);checks={}
    if variant.startswith('finite'):
        with torch.no_grad():
            for i in [0,1,2,3,4,5,18,19,21,22,23]:
                pos,c=draws(initial[i].count,gen,8)
                dense=first.nonlinear_chunk(model,ctxs[0],i,initial[i].read[:,c],initial[i].write[:,c])
                observed=score(model,ctxs[0],i,initial[i],pos,c,variant)
                expected=dense[pos,torch.arange(len(c),device='cuda')]
                torch.testing.assert_close(observed,expected,atol=2e-4,rtol=2e-3)
                checks[MODULES[i]]=float((observed-expected).abs().max())
    refs=reference(model,ctxs,initial,variant)
    a,opt=create(initial);update(model,ctxs,a,opt,refs,cfg,variant,0)
    # Round-trip the actual checkpoint format: Adam's CPU step tensor would
    # otherwise alias the live optimizer through detach().cpu().
    import io
    buffer=io.BytesIO();torch.save(cpu_tree(state(a,opt,refs,1)),buffer);buffer.seek(0)
    saved=torch.load(buffer,map_location='cpu',weights_only=False)
    update(model,ctxs,a,opt,refs,cfg,variant,1)
    b,bopt=create(initial,saved);update(model,ctxs,b,bopt,refs,cfg,variant,1)
    for x,y in zip(a,b):
        torch.testing.assert_close(x.read,y.read,rtol=0,atol=0);torch.testing.assert_close(x.write,y.write,rtol=0,atol=0)
    errors=[core.reconstruction_error(model.get_submodule(n).weight,x) for n,x in zip(MODULES,a)]
    assert max(errors)<1e-5
    # Two independent estimates on fixed factors quantify sampling noise.
    update(model,ctxs,a,opt,refs,cfg,variant,0,sampling_seed=7001,apply_step=False)
    ga=[t.grad.detach().clone() for x in a for t in [x.read,x.write]]
    update(model,ctxs,a,opt,refs,cfg,variant,0,sampling_seed=7002,apply_step=False)
    gb=[t.grad for x in a for t in [x.read,x.write]]
    na=sum(float(x.square().sum()) for x in ga);nb=sum(float(x.square().sum()) for x in gb)
    cosine=sum(float((x*y).sum()) for x,y in zip(ga,gb))/math.sqrt(na*nb)
    difference=math.sqrt(sum(float((x-y).square().sum()) for x,y in zip(ga,gb))/max((na+nb)/2,1e-30))
    del ga,gb
    timings=[]
    for step in range(2,5):
        start=time.perf_counter();fresh=caches(model,batch(training,step),variant,step)
        update(model,fresh,a,opt,refs,cfg,variant,step);torch.cuda.synchronize()
        timings.append(time.perf_counter()-start);del fresh
    del a,b,opt,bopt,ctxs,saved
    import gc
    gc.collect();torch.cuda.empty_cache()
    control_cache=capture_gim(model,rows)
    index=torch.randperm(32768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    control_ref=[method.slice_entry(e,index) for e in control_cache];del control_cache
    ticks=[];last=time.perf_counter()
    def mark(_):
        nonlocal last
        torch.cuda.synchronize();now=time.perf_counter();ticks.append(now-last);last=now;return False
    fitted,_=method.fit(control_ref,[model.get_submodule(n).weight for n in MODULES],initial,cfg,
        stop_after=4,batch_provider=lambda s:capture_gim(model,batch(training,s)),step_callback=mark,log=lambda _:None)
    del fitted
    baseline=st.median(ticks[1:])
    record=dict(passed=True,sample_dense_checks=checks,resume_bitwise=True,max_matrix_error=max(errors),
                seconds_per_step=st.median(timings),samples_per_context=SAMPLES,training_contexts=64,
                projected_5000_seconds=5000*st.median(timings),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                two_sample_gradient_cosine=cosine,two_sample_gradient_relative_difference=difference,
                matched_gim_seconds_per_step=baseline,relative_to_gim=st.median(timings)/baseline)
    save_json(record,out/'prepare.json');checkpoint(refs,out/'references.pt')
    print('PREPARE_COMPLETE '+json.dumps(record),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train']);ap.add_argument('variant',choices=VARIANTS);args=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID');credentials();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed();torch.manual_seed(0)
    out=ROOT/'output'/EXPERIMENT/args.variant;out.mkdir(parents=True,exist_ok=True)
    control=json.loads((ROOT/'output/gim_1h/plan.json').read_text());cfg=method.TrainConfig(**control['training_config'])
    training,manifest=tokens();model=target();counts=[control['component_counts'][n] for n in MODULES]
    initial=[core.initialize_pieces(model.get_submodule(n).weight,c,0) for n,c in zip(MODULES,counts)]
    plan=dict(variant=args.variant,updates=5000,training_config=asdict(cfg),batch_contexts=64,samples_per_context=SAMPLES,
              target_sha256=TARGET_SHA,data_sha256=manifest['train']['sha256'],source_sha256=sha(__file__),
              component_counts=counts,initialization='Fresh seed-0, matching GIM; no VPD warm start',
              mask_free_training=True,primary_metric='Actual KLKeep at fixed L0 with common Fisher K64 ranking',
              finite_probe='GIM CE' if args.variant=='finite_gim' else 'one independent model-label sample per output position' if args.variant=='finite_fisher' else None)
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==json.loads(json.dumps(plan))
    else:save_json(plan,out/'plan.json')
    if args.mode=='prepare':prepare(model,training,initial,cfg,args.variant,out);return
    assert json.loads((out/'prepare.json').read_text())['passed']
    saved=None
    if (out/'latest.json').exists():saved=torch.load(json.loads((out/'latest.json').read_text())['checkpoint'],map_location='cpu',weights_only=False)['state']
    pieces,opt=create(initial,saved);refs=torch.load(out/'references.pt',map_location='cuda',weights_only=False)
    start=saved['next_step'] if saved else 0;del saved,initial
    if start>=5000:return
    previous=json.loads((out/'wandb.json').read_text()) if (out/'wandb.json').exists() else None
    run=wandb.init(project='param-autoenc',entity='rohan-kathuria-neu',group=EXPERIMENT,name=args.variant+'-5000',
                   dir=str(out),config=plan,id=previous['id'] if previous else None,resume='must' if previous else None)
    save_json(dict(id=run.id,url=run.url),out/'wandb.json')
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGUSR1,stopping);signal.signal(signal.SIGTERM,stopping)
    deadline=float(os.environ.get('OBJECTIVE_DEADLINE','inf'));began=time.perf_counter();completed=start
    def save():
        dest=out/'checkpoints'/f'{completed:06d}'
        identity=dict(method='parauto',sensor=args.variant,completed=completed,training_positions=completed*32768,
                      target_sha256=TARGET_SHA,data_sha256=manifest['train']['sha256'])
        checkpoint(dict(**identity,state=state(pieces,opt,refs,completed)),dest/'checkpoint.pt')
        checkpoint(dict(**identity,factors=[dict(read=p.read,write=p.write) for p in pieces],provenance=plan),dest/'model.pth')
        save_json(dict(step=completed,checkpoint=str(dest/'checkpoint.pt'),model=str(dest/'model.pth')),out/'latest.json')
    for step in range(start,5000):
        ctxs=caches(model,batch(training,step),args.variant,step)
        metrics=update(model,ctxs,pieces,opt,refs,cfg,args.variant,step);del ctxs;completed=step+1
        if completed%25==0:
            metrics.update(update=completed,positions=completed*32768,seconds=time.perf_counter()-began,
                           seconds_per_step=(time.perf_counter()-began)/(completed-start))
            save_json(metrics,out/'progress.json');run.log(metrics,step=completed);print(json.dumps(metrics),flush=True)
        if completed%1000==0 or completed in [250,2500] or stop or time.time()>deadline:
            save()
            errors=[core.reconstruction_error(model.get_submodule(n).weight,p) for n,p in zip(MODULES,pieces)]
            assert max(errors)<1e-5
        if stop or time.time()>deadline:break
    if completed==5000:
        save_json(dict(complete=True,completed=completed),out/'training_complete.json')
        modelpath=Path(json.loads((out/'latest.json').read_text())['model'])
        upload(modelpath,f'experiments/{EXPERIMENT}/{args.variant}/model_5000.pth')
        for p in [out/'plan.json',out/'prepare.json',out/'training_complete.json']:
            upload(p,f'experiments/{EXPERIMENT}/{args.variant}/{p.name}')
    run.finish();print('TRAINING_STOP '+str(completed),flush=True)
    if completed<5000:raise SystemExit(75)


if __name__=='__main__':main()
