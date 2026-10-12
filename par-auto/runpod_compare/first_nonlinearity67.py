"""Exact immediate GELU/softmax/RMS, then cached derivatives. Explorer 67M."""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/objective67_20260915'))
import objective67 as obj
audit, core = obj.audit, obj.core
import numpy as np
import torch
import torch.nn.functional as F
from common import MODULES, TARGET_SHA, target, tokens, save_json, sha, upload, credentials
from gim_attribution import gim_context, ScaleGradient, capture_gim

EXPERIMENT = 'first_nonlinearity67_20260915'
OUT = ROOT/'output'/EXPERIMENT
CHUNK = 16
POWER = .641
ROWS = list(range(464,472))


def capture(model, rows, sensor):
    """One FP32 forward/backward, retaining downstream cotangents explicitly."""
    pre, post, norms, probabilities, handles = {}, {}, {}, [], []
    names = [f'h.{i}.rms_{j}' for i in range(4) for j in [1,2]]+['ln_f']
    with gim_context(model, temperature=2. if sensor=='gim' else 1., freeze_rms=sensor=='gim'):
        softmax = F.softmax
        def record_softmax(x, *args, **kwargs):
            p = softmax(x,*args,**kwargs)
            if x.ndim==4: probabilities.append((x,p))
            return p
        F.softmax = record_softmax
        def matrix_hook(name):
            def hook(m, inputs, output):
                pre[name],post[name] = inputs[0],output
                if sensor=='gim':
                    scale = {'q_proj':.25,'k_proj':.25,'v_proj':.5}.get(name.split('.')[-1])
                    if scale is not None: return ScaleGradient.apply(output,scale)
            return hook
        def norm_hook(name):
            def hook(m,inputs,output): norms[name]=(inputs[0],output)
            return hook
        try:
            for n in MODULES: handles.append(model.get_submodule(n).register_forward_hook(matrix_hook(n)))
            for n in names: handles.append(model.get_submodule(n).register_forward_hook(norm_hook(n)))
            handles.append(model.wte.register_forward_hook(lambda m,i,o:o.detach().requires_grad_()))
            logits = model(rows[:,:-1])[0]
            loss = F.cross_entropy(logits.flatten(0,1),rows[:,1:].flatten(),reduction='sum')
            assert len(probabilities)==4
            nodes = [post[n] for n in MODULES]+[norms[n][1] for n in names]+[p for _,p in probabilities]
            grads = torch.autograd.grad(loss,nodes)
        finally:
            for h in handles: h.remove()
            F.softmax = softmax
    g = {n:v[0].detach() for n,v in zip(MODULES,grads[:24])}
    ng = {n:v[0].detach() for n,v in zip(names,grads[24:33])}
    result = dict(sensor=sensor,rows=rows,base=logits.detach(),
                  inputs={n:x[0].detach() for n,x in pre.items()},
                  outputs={n:x[0].detach() for n,x in post.items()},g=g,
                  norms={n:(x[0].detach(),y[0].detach(),ng[n]) for n,(x,y) in norms.items()},att={})
    with torch.no_grad():
        for i,((l,p),bg) in enumerate(zip(probabilities,grads[33:])):
            a=model.h[i].attn;T=rows.shape[1]-1
            q=result['outputs'][MODULES[i*6]].reshape(T,6,128).transpose(0,1)[None]
            k=result['outputs'][MODULES[i*6+1]].reshape(T,6,128).transpose(0,1)[None]
            q,k=a.apply_rotary_pos_emb(q,k,a.rotary_cos[:T][None],a.rotary_sin[:T][None])
            l,p,b=l[0].detach(),p[0].detach(),bg[0].detach()
            # Exclusive sums avoid subtracting a dominant key from the row sum.
            w=(l.double()-l.double().amax(-1,keepdim=True)).exp()
            def exclusive(x):
                left=F.pad(x.cumsum(-1)[...,:-1],(1,0))
                right=F.pad(x.flip(-1).cumsum(-1).flip(-1)[...,1:],(0,1))
                return left+right
            rest=exclusive(w)
            other=(exclusive(w*b.double())/rest.clamp_min(1e-300)).float()
            logrest=(rest.log()+l.double().amax(-1,keepdim=True)).float()
            result['att'][i]=dict(q=q[0],k=k[0],logits=l,p=p,b=b,other=other,logrest=logrest)
    return result


def rotated(model, layer, write, T):
    att=model.h[layer].attn
    u=write.T.reshape(-1,6,128).permute(1,0,2)[:,None]
    return u*att.rotary_cos[:T][None,:,None]+att.rotate_every_two(u)*att.rotary_sin[:T][None,:,None]


def nonlinear_chunk(model,ctx,index,read,write):
    name=MODULES[index];kind=index%6;layer=index//6
    a=ctx['inputs'][name]@read
    linear=-a*(ctx['g'][name]@write)
    if kind==2:
        # V has no immediately adjacent nonlinearity. Keep the original score.
        return linear
    if kind==4:
        x=ctx['outputs'][name]
        down=model.get_submodule(MODULES[index+1])
        b=ctx['g'][MODULES[index+1]]@down.weight
        d=F.gelu(x[:,None]-a[:,:,None]*write.T[None],approximate='tanh')-F.gelu(x,approximate='tanh')[:,None]
        return (d*b[:,None]).sum(-1)
    if kind in [3,5]:
        n=f'h.{layer}.rms_2' if kind==3 else (f'h.{layer+1}.rms_1' if layer<3 else 'ln_f')
        r,_,b=ctx['norms'][n];norm=model.get_submodule(n)
        s2=r.square().mean(-1,keepdim=True)+norm.variance_epsilon
        inv=s2.rsqrt();ru=r@write;u2=write.square().sum(0)
        new=(s2+(-2*a*ru+a.square()*u2)/r.shape[-1]).clamp_min(norm.variance_epsilon).rsqrt()
        br=(b*norm.weight*r).sum(-1,keepdim=True)
        bu=(b*norm.weight)@write
        finite=(new-inv)*br-a*new*bu
        jdelta=-a*inv*bu
        if ctx['sensor']=='true':jdelta=jdelta+a*inv*ru/(r.shape[-1]*s2)*br
        return linear+finite-jdelta
    att=ctx['att'][layer];T=a.shape[0];u=rotated(model,layer,write,T)
    factor=.25 if ctx['sensor']=='gim' else 1.
    if kind==0:
        d=-torch.einsum('htcd,hsd->htcs',u,att['k'])*a[None,:,:,None]/math.sqrt(128)
        p=(att['logits'][:,:,None]+d).softmax(-1)
        return factor*((p-att['p'][:,:,None])*att['b'][:,:,None]).sum(-1).sum(0)
    assert kind==1
    d=-torch.einsum('htd,hscd->htsc',att['q'],u)*a[None,None]/math.sqrt(128)
    l=att['logits'];rest=att['logrest'];valid=torch.isfinite(l)&torch.isfinite(rest)
    odds=torch.where(valid,l-rest,torch.zeros_like(l))
    dp=(odds[:,:,:,None]+d).sigmoid()-odds.sigmoid()[:,:,:,None]
    value=dp*(att['b']-att['other'])[:,:,:,None]*valid[:,:,:,None]
    return factor*value.sum(0).sum(0)


def penalty(x):return (x.abs()+1e-6).pow(POWER)-1e-6**POWER


def evaluate(model,ctx,pieces,backward=False,scales=None,linear_only=False):
    result=[]
    for i,p in enumerate(pieces):
        total=0.
        for start in range(0,p.count,CHUNK):
            read,write=p.read[:,start:start+CHUNK],p.write[:,start:start+CHUNK]
            if linear_only:
                s=-(ctx['inputs'][MODULES[i]]@read)*(ctx['g'][MODULES[i]]@write)
            else:s=nonlinear_chunk(model,ctx,i,read,write)
            loss=penalty(s).sum(-1).mean()
            assert torch.isfinite(loss)
            if backward:(loss/(24*scales[i])).backward()
            total+=float(loss.detach())
        result.append(total)
    return result


@torch.no_grad()
def validate(model,ctx,pieces):
    checks={}
    # Compare corrected sites against actual target forwards with one source ablated.
    for index in [0,1,3,4,5,18,19,21,22,23]:
        name=MODULES[index];p=pieces[index];source=127
        strength=(ctx['inputs'][name][source]@p.read).abs()*p.write.norm(dim=0)
        c=int(strength.argmax());read=p.read[:,c:c+1];write=p.write[:,c:c+1]
        predicted=float(nonlinear_chunk(model,ctx,index,read,write)[source,0])
        z=(ctx['inputs'][name][source]@read)*write[:,0]
        got={};handles=[];softmax=F.softmax;layer=index//6;kind=index%6;calls=0
        def remove(m,args,out):
            out=out.clone();out[0,source]-=z;return out
        def collect_softmax(x,*args,**kwargs):
            nonlocal calls
            out=softmax(x,*args,**kwargs)
            if x.ndim==4:
                if calls==layer:got['p']=out[0]
                calls+=1
            return out
        def norm_input(m,args,out):got['norm']=out[0]
        def gelu_output(m,args):got['gelu']=args[0][0]
        try:
            handles.append(model.get_submodule(name).register_forward_hook(remove));F.softmax=collect_softmax
            if kind in [3,5]:
                n=f'h.{layer}.rms_2' if kind==3 else (f'h.{layer+1}.rms_1' if layer<3 else 'ln_f')
                handles.append(model.get_submodule(n).register_forward_hook(norm_input))
            elif kind==4:handles.append(model.get_submodule(MODULES[index+1]).register_forward_pre_hook(gelu_output))
            model(ctx['rows'][:,:-1])
        finally:
            F.softmax=softmax
            for h in handles:h.remove()
        if kind in [0,1]:
            att=ctx['att'][layer];expected=float(((got['p']-att['p'])*att['b']).sum())
        elif kind==4:
            b=ctx['g'][MODULES[index+1]]@model.get_submodule(MODULES[index+1]).weight
            expected=float(((got['gelu']-F.gelu(ctx['outputs'][name],approximate='tanh'))*b).sum())
        else:
            r,base,b=ctx['norms'][n];norm=model.get_submodule(n)
            rr=r[source];inv=(rr.square().mean()+norm.variance_epsilon).rsqrt()
            jz=norm.weight*(inv*z-inv**3*rr*(rr*z).mean())
            expected=float(-ctx['g'][name][source]@z+((got['norm']-base)*b).sum()+b[source]@jz)
        error=abs(predicted-expected)
        checks[name]=dict(predicted=predicted,explicit=expected,absolute_error=error)
        assert error<2e-4+2e-3*abs(expected),(name,predicted,expected)
    return checks


def benchmark(model,rows,ctx,pieces,scales):
    p=obj.clone(pieces,grad=True)
    def run(linear):
        for x in p:x.read.grad=None;x.write.grad=None
        evaluate(model,ctx,p,backward=True,scales=scales,linear_only=linear)
    result={}
    for label,linear in [('linear',True),('finite',False)]:
        run(linear)
        values=[obj.timed(lambda:run(linear))[1] for _ in range(3)]
        result[label]=st.median(values)
        assert all(torch.isfinite(x.read.grad).all() and torch.isfinite(x.write.grad).all() for x in p)
    # Established BF16 GIM reference, with warmed cache and factor backward.
    capture_gim(model,rows,microbatch=1)
    caches=[obj.timed(lambda:capture_gim(model,rows,microbatch=1)) for _ in range(3)]
    cache=caches[-1][0];gctx=dict(cache={'gim':cache})
    oldscale=json.loads((ROOT/'output/objective67_20260915/normalizers.json').read_text())
    def gim_run():
        loss=obj.reduce_scores(obj.scores(model,gctx,p,'gim'),'gim',POWER,oldscale)
        torch.autograd.grad(loss,[t for x in p for t in [x.read,x.write]])
    gim_run();bt=st.median(obj.timed(gim_run)[1] for _ in range(3))
    result['gim_bf16_capture']=st.median(t for _,t in caches);result['gim_bf16_backward']=bt
    result['gim_bf16_total']=result['gim_bf16_capture']+bt
    return result


def summarize(records,scales,timing):
    summary={}
    for sensor in ['true','gim']:
        summary[sensor]={}
        for variant in ['linear','finite']:
            means={name:st.mean(sum(r['states'][name][sensor][variant][i]/scales[sensor][i] for i in range(24))/24 for r in records) for name in records[0]['states']}
            key=lambda r,n:sum(r['states'][n][sensor][variant][i]/scales[sensor][i] for i in range(24))/24
            summary[sensor][variant]=dict(means=means,published_vpd_over_gim=means['vpd_published']/means['gim5k'],
                exact_vpd_over_gim=means['vpd_exact']/means['gim5k'],
                published_preferred_fraction=st.mean(key(r,'vpd_published')<key(r,'gim5k') for r in records),
                damage_increases_fraction=st.mean(key(r,b)>key(r,a) for r in records for a,b in zip(['vpd_exact','descent_20','descent_100'],['descent_20','descent_100','descent_200'])),
                by_kind={MODULES[i].split('.')[-1]:{n:st.mean(sum(r['states'][n][sensor][variant][j]/scales[sensor][j] for j in range(i,24,6))/4 for r in records) for n in ['vpd_published','vpd_exact','gim5k']} for i in range(6)})
    save_json(dict(summary=summary,scales=scales,timing=timing,records=records),OUT/'comparison.json')
    lines=['# Exact first-nonlinearity screen','',
           'Frozen 67M target on Explorer; eight fresh contexts, no factor updates. Lower objective for VPD is the desired ordering.', '',
           '| Downstream derivative | Sensor | Published VPD/GIM | Exact VPD/GIM | Published preferred fraction | Damage raises loss |',
           '|---|---|---:|---:|---:|---:|']
    for sensor,variants in summary.items():
        for variant,v in variants.items():lines.append(f"| {sensor} | {variant} | {v['published_vpd_over_gim']:.5f} | {v['exact_vpd_over_gim']:.5f} | {v['published_preferred_fraction']:.3f} | {v['damage_increases_fraction']:.3f} |")
    lines+=['','Exact first nonlinearity: Q/K softmax, MLP-in GELU, O/down RMS with the residual bypass preserved. '
            'V remains the linear control: its path to a norm crosses attention mixing and O, rather than an immediately adjacent nonlinearity. '
            'All components and all 512 source positions are scored, with fixed exponent 0.641. '
            'Each module uses a shared GIM5k linear-score training-context normalizer for both sensor variants and all checkpoints. '
            'Caches are FP32. The true cache uses ordinary CE derivatives; GIM uses its surrogate rules. '
            'These scalar probes are not a complete KL metric. No CI or support enters the objective.','',
            'Per-kind results, validation, raw context scores, and timing are in JSON. Timing includes factor backward but excludes optimizer/projection; '
            'the dense finite-GELU implementation is chunked and not claimed to fit the GIM production budget.','']
    for sensor in ['true','gim']:
        t=timing[sensor]
        lines.append(f"{sensor}: finite capture + backward {t['finite_total']:.4f} seconds; {t['relative_to_gim']:.2f} times the warmed BF16 GIM reference.")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed();OUT.mkdir(parents=True,exist_ok=True)
    model=target();train,manifest=tokens('train');val,_=tokens('val')
    gim,_,gm=audit.shared.load(model,'gim_5000');old,trainer,vm=audit.shared.load(model,'vpd_published')
    states=dict(vpd_published=old,vpd_exact=obj.read_pieces(ROOT/'output/gap67_bridge_20260915/bridge_model.pth'),gim5k=gim)
    del trainer
    for step in [20,100,200]:states[f'descent_{step}']=obj.read_pieces(ROOT/f'output/gap67_descent_20260915/{step}.pth')
    save_json(dict(job=os.environ['SLURM_JOB_ID'],source_sha256=sha(__file__),target_sha256=TARGET_SHA,data=manifest,
                   sources=dict(gim=gm,vpd=vm),counts={n:[p.count for p in ps] for n,ps in states.items()},
                   confirmation=ROWS,calibration=[10000],power=POWER,chunk=CHUNK,
                   note='Full deterministic component sums. No factor updates. V retains linear attribution.'),OUT/'plan.json')
    rows=torch.tensor(np.array(train[10000:10001],dtype=np.int64),device='cuda')
    scales,timing={},{}
    for sensor in ['true','gim']:
        ctx=capture(model,rows,sensor)
        if sensor=='true':save_json(validate(model,ctx,states['vpd_exact']),OUT/'validation.json')
        with torch.no_grad():scales[sensor]=[max(x,1e-10) for x in evaluate(model,ctx,gim,linear_only=True)]
        print('PREFLIGHT '+sensor,flush=True)
        timing[sensor]=benchmark(model,rows,ctx,gim,scales[sensor])
        timing[sensor]['capture_seconds']=st.median(obj.timed(lambda:capture(model,rows,sensor))[1] for _ in range(3))
        timing[sensor]['finite_total']=timing[sensor]['capture_seconds']+timing[sensor]['finite']
        timing[sensor]['relative_to_gim']=timing[sensor]['finite_total']/timing[sensor]['gim_bf16_total']
        save_json(timing,OUT/'timing.json');del ctx
    save_json(scales,OUT/'scales.json');records=[]
    for seq in ROWS:
        record=dict(sequence=seq,states={n:{} for n in states})
        rows=torch.tensor(np.array(val[seq:seq+1],dtype=np.int64),device='cuda')
        for sensor in ['true','gim']:
            ctx=capture(model,rows,sensor)
            with torch.no_grad():
                for name,pieces in states.items():
                    record['states'][name][sensor]={}
                    for variant in ['linear','finite']:
                        values,seconds=obj.timed(lambda:evaluate(model,ctx,pieces,linear_only=variant=='linear'))
                        record['states'][name][sensor][variant]=values
                        print(json.dumps(dict(sequence=seq,sensor=sensor,state=name,variant=variant,
                                              loss=st.mean(v/s for v,s in zip(values,scales[sensor])),seconds=seconds)),flush=True)
            del ctx
        save_json(record,OUT/f'{seq}.json');records.append(record)
        summarize(records,scales,timing)
    receipts=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file():receipts.append(upload(p,f'experiments/{EXPERIMENT}/{p.name}'))
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.is_file():receipts.append(upload(p,f'experiments/{EXPERIMENT}/source/{p.name}'))
    save_json(dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(OUT/'comparison.json')),OUT/'complete.json')
    upload(OUT/'complete.json',f'experiments/{EXPERIMENT}/complete.json')
    save_json(dict(complete=True),OUT/'analysis_complete.json')
    print('FIRST_NONLINEARITY67_COMPLETE',flush=True)


if __name__=='__main__':main()
