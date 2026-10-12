"""Frozen 67M IFR-style and signed reader-conditioned selection; Explorer only.

This is NOT a reproduction of IFR's head-level graph extraction. All 24 parameter
matrices are scored; no factors are trained and no KL/gradient is consulted for
selection. Reader conditioning is a bounded one-hop approximation, not a proof
of complete routes or finite sufficiency. See ifr67.md for exact definitions.
"""
import json
import math
import os
from pathlib import Path
import statistics as st
import sys
import time

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/gap67_bridge_20260915'))
import gap67_bridge as bridge_tools
from gap67_bridge import audit
import numpy as np
import torch
from common import MODULES, TARGET_SHA, target, tokens, save_json, sha, credentials, upload
from train_pair import core
from fisher_attribution import capture_fisher
from gim_attribution import capture_gim

OUT = ROOT/'output/ifr67_20260915'
FIRST, SEQUENCES = 432, 8
READERS, DESTINATIONS, ROUNDS = 16, 4, 2


def timed(fn):
    torch.cuda.synchronize(); start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter()-start


def normalize(s):
    return s/s.sum(-1, keepdim=True).clamp_min(1e-20)


def finite_scores(hread, write, y, directions=None, *, signed=False):
    """Finite L1 proximity, exact in FP32; stream tensors instead of T*C*D storage.

    directions[t,r,d] projects a writer's output into consumer read coordinates.
    y is the complete observed consumer read vector in that case.
    """
    result = torch.empty_like(hread)
    for t in range(0, len(hread), 16):
        yy = y[t:t+16]
        norm = yy.abs().sum(-1, keepdim=True)
        for c in range(0, write.shape[1], 512):
            w = write[:, c:c+512]
            if directions is None:
                z = hread[t:t+16, c:c+512, None]*w.T[None]
            else:
                projected = (directions[t:t+16] @ w).transpose(-1, -2)
                z = hread[t:t+16, c:c+512, None]*projected
            delta = norm-(yy[:, None]-z).abs().sum(-1)
            result[t:t+16, c:c+512] = delta.abs() if signed else delta.clamp_min(0)
    return result


@torch.no_grad()
def capture(model, ids):
    inputs, outputs, hooks = {}, {}, []
    names = list(MODULES)+[f'h.{i}.rms_{j}' for i in range(4) for j in [1,2]]
    try:
        for name in names:
            def hook(_module, args, output, name=name):
                inputs[name] = args[0][0].detach().float()
                outputs[name] = output[0].detach().float()
            hooks.append(model.get_submodule(name).register_forward_hook(hook))
        logits = model(ids)[0]
    finally:
        for hook in hooks: hook.remove()
    attention, checks = {}, {}
    for i in range(4):
        name = f'h.{i}.attn'
        att = model.get_submodule(name)
        assert att.n_head == att.n_key_value_heads == 6
        assert att.head_dim == 128
        T = ids.shape[1]
        q = outputs[name+'.q_proj'].reshape(1,T,6,128).transpose(1,2)
        k = outputs[name+'.k_proj'].reshape(1,T,6,128).transpose(1,2)
        q,k = att.apply_rotary_pos_emb(q,k,att.rotary_cos[:T][None],att.rotary_sin[:T][None])
        scores = (q @ k.transpose(-1,-2))/math.sqrt(128)
        scores.masked_fill_(~torch.ones(T,T,device='cuda',dtype=torch.bool).tril(), -float('inf'))
        alpha = scores.softmax(-1)[0]
        v = outputs[name+'.v_proj'].reshape(T,6,128).transpose(0,1)
        reproduced = (alpha @ v).transpose(0,1).reshape(T,768)
        err = float((reproduced-inputs[name+'.o_proj']).norm()/inputs[name+'.o_proj'].norm())
        assert err < 1e-5, err
        attention[i] = alpha
        checks[name+'_attention_relative_error'] = err
    return logits, inputs, outputs, attention, checks


@torch.no_grad()
def local_scores(model, pieces, inputs):
    positive, signed, hr = [], [], []
    for name,p in zip(MODULES,pieces,strict=True):
        layer = model.get_submodule(name)
        assert layer.bias is None  # match additive parameter-piece decomposition
        h = inputs[name]
        a = h @ p.read
        y = h @ layer.weight.T
        # Calculate signed delta once; positive and absolute variants share it.
        pos = finite_scores(a,p.write,y,signed=False)
        absolute = finite_scores(a,p.write,y,signed=True)
        positive.append(normalize(pos)); signed.append(normalize(absolute)); hr.append(a)
    return positive, signed, hr


def selected_readers(index, keep_parts, local, pieces):
    p = pieces[index]
    keep = keep_parts[index]
    order = local[index].masked_fill(~keep, -float('inf')).topk(READERS, -1).indices
    valid = keep.gather(-1, order)
    dirs = p.read.T[order]
    dirs = dirs/dirs.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    dirs = dirs*valid[...,None]
    return dirs, dict(selected=int(keep.sum()), covered=int(valid.sum()))


def validate_algebra():
    gen = torch.Generator(device='cuda').manual_seed(915)
    a = torch.randn(5,19,device='cuda',generator=gen)
    w = torch.randn(11,19,device='cuda',generator=gen)
    d = torch.randn(5,7,11,device='cuda',generator=gen)
    y = torch.randn(5,7,device='cuda',generator=gen)
    z = torch.einsum('tc,dc,trd->tcr',a,w,d)
    delta = y.abs().sum(-1)[:,None]-(y[:,None]-z).abs().sum(-1)
    for signed in [False,True]:
        observed = finite_scores(a,w,y,d,signed=signed)
        expected = delta.abs() if signed else delta.clamp_min(0)
        torch.testing.assert_close(observed,expected,rtol=1e-5,atol=1e-5)
        scale = torch.linspace(.2,3,19,device='cuda')
        torch.testing.assert_close(finite_scores(a*scale,w/scale,y,d,signed=signed),observed,rtol=1e-5,atol=1e-5)
    aa = torch.tensor([[.6,.8,-.4]],device='cuda')
    ww = torch.ones(1,3,device='cuda'); yy = torch.ones(1,1,device='cuda')
    torch.testing.assert_close(finite_scores(aa,ww,yy),torch.tensor([[.6,.8,0]],device='cuda'))
    torch.testing.assert_close(finite_scores(aa,ww,yy,signed=True),aa.abs())
    return dict(projected_proximity=True,gauge_invariance=True,negative_cancellation=True)


@torch.no_grad()
def condition(model, pieces, keep, local, hr, inputs, outputs, attention):
    parts = keep.split([p.count for p in pieces],-1)
    readers, coverage = {}, {}
    for i in range(len(pieces)):
        readers[i],coverage[MODULES[i]] = selected_readers(i,parts,local,pieces)
    scores, diagnostics = [], {}
    for i,(name,p) in enumerate(zip(MODULES,pieces,strict=True)):
        layer, kind = i//6, i%6
        d, y = None, None
        if kind == 2:  # V(source) -> attention mixing -> selected O(destination) reads
            alpha = attention[layer]
            # Bounded destination shortlist for each source token. Preserve every
            # head's signed writer/read contraction within each chosen route.
            dest = alpha.mean(0).T.topk(DESTINATIONS,-1).indices  # [source,destination]
            r = readers[i+1][dest]  # [source,destination,reader,dim]
            a = alpha.permute(2,1,0).gather(1,dest[...,None].expand(-1,-1,6))
            d = (r*a.repeat_interleave(128,-1)[:,:,None,:]).flatten(1,2)
            consumer = inputs[MODULES[i+1]][dest]
            y = (r*consumer[:,:,None,:]).sum(-1).flatten(1,2)
            diagnostics[name+'_destination_attention_mass'] = float(a.sum(1).mean())
        elif kind == 3:  # O -> residual addition -> observed RMS -> FC reads
            norm = model.get_submodule(f'h.{layer}.rms_2')
            residual = inputs[f'h.{layer}.rms_2']
            scale = torch.rsqrt(residual.square().mean(-1,keepdim=True)+norm.variance_epsilon)*norm.weight
            r = readers[i+1]
            d = r*scale[:,None]
            y = (r*inputs[MODULES[i+1]][:,None]).sum(-1)
        elif kind == 4:  # FC -> observed GELU secant -> Down reads
            x = outputs[name]
            secant = .5*(1+torch.tanh(math.sqrt(2/math.pi)*(x+.044715*x.pow(3))))
            torch.testing.assert_close(x*secant,inputs[MODULES[i+1]],rtol=2e-5,atol=2e-6)
            r = readers[i+1]
            d = r*secant[:,None]
            y = (r*inputs[MODULES[i+1]][:,None]).sum(-1)
        elif kind == 5 and layer < 3:  # Down -> residual -> next block Q/K/V reads
            norm = model.get_submodule(f'h.{layer+1}.rms_1')
            residual = inputs[f'h.{layer+1}.rms_1']
            scale = torch.rsqrt(residual.square().mean(-1,keepdim=True)+norm.variance_epsilon)*norm.weight
            r = torch.cat([readers[i+j] for j in [1,2,3]],1)
            d = r*scale[:,None]
            y = (r*inputs[MODULES[i+1]][:,None]).sum(-1)
        if d is None:
            # Q/K remain fully charged to the budget and locally scored. This
            # first prototype does not attribute creation of attention routes.
            scores.append(local[i]); continue
        routed = normalize(finite_scores(hr[i],p.write,y,d,signed=True))
        active = routed.sum(-1,keepdim=True) > 0
        # Prespecified blend avoids suppressing branches absent from the current
        # reader shortlist; no hyperparameter is chosen using evaluation KL.
        mixed = torch.where(active,.5*local[i]+.5*routed,local[i])
        scores.append(mixed)
        diagnostics[name+'_route_nonzero_tokens'] = float(active.float().mean())
    return torch.cat(scores,-1),dict(reader_coverage=coverage,diagnostics=diagnostics)


def summarize(records, plan):
    result = dict(plan=plan, summaries={})
    lines = ['# IFR-style selection on the frozen 67M model','',
             'No factor training. Eight fresh contexts 432–439. FP32 KL with attention and nonlinearities recomputed. '
             'Reader-conditioned selection is a bounded, signed, one-hop extension, not full IFR or a causal guarantee.','',
             '| Factors | Budget | Selector | KL | Restore last KL | Clean-error injection KL |',
             '|---|---|---|---:|---:|---:|']
    for family in ['gim','vpd_exact']:
        for budget in ['native','200','512']:
            rr = [r for r in records if r['family']==family and r['budget']==budget]
            if not rr: continue
            result['summaries'][family+'_'+budget] = {}
            for label in rr[0]['variants']:
                vv = [r['variants'][label] for r in rr]
                stats = {k:st.mean(v[k] for v in vv) for k in ['kl','restore_last_kl','clean_error_injection_kl','count_mean']}
                if len(vv)>1: stats['kl_context_se'] = st.stdev(v['kl'] for v in vv)/math.sqrt(len(vv))
                result['summaries'][family+'_'+budget][label] = stats
                lines.append(f"| {family} | {budget} | {label} | {stats['kl']:.5f} | {stats['restore_last_kl']:.5f} | {stats['clean_error_injection_kl']:.5f} |")
    lines += ['', 'Reader-conditioned rounds are reported separately; no round, initialization, or hyperparameter '
              'is selected by KL. Native CI is used only for the witness and the matched-count control. '
              'Q/K use local scores, while V→O, O→FC, FC→Down, and Down→next Q/K/V receive reader conditioning. '
              'Read shortlists cap at 16 per consumer/token and 4 attention destinations/source. '
              'Signs enter contractions and the finite proximity; absolute proximity credits both positive and negative effects.','']
    save_json(result,OUT/'comparison.json'); (OUT/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed(); OUT.mkdir(parents=True,exist_ok=True)
    save_json(validate_algebra(),OUT/'algebra_validation.json')
    model = target(); data,manifest = tokens('val')
    assert sum(p.numel() for p in model.parameters()) == 66931200
    old,trainer,oldmeta = audit.shared.load(model,'vpd_published')
    ours,_,oursmeta = audit.shared.load(model,'gim_5000')
    path = ROOT/'output/gap67_bridge_20260915/bridge_model.pth'
    saved = torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert saved['target_sha256']==TARGET_SHA
    exact = [core.FreePieces(p['read'].cuda(),p['write'].cuda()) for p in saved['factors']]
    del saved
    assert all(a.count==b.count for a,b in zip(exact,ours))
    errors = [core.reconstruction_error(model.get_submodule(n).weight,p) for n,p in zip(MODULES,exact)]
    assert max(errors)<1e-5
    plan = dict(source_sha256=sha(__file__), target_sha256=TARGET_SHA,val_sha256=manifest['val']['sha256'],
                first=FIRST,sequences=SEQUENCES,readers=READERS,destinations=DESTINATIONS,rounds=ROUNDS,
                budgets=['native',200,512],gim=oursmeta,vpd=oldmeta,bridge_sha256=sha(path),
                precision='FP32; TF32 disabled; Fisher/GIM cache BF16', factor_training=False,
                selection_uses_gradients=False,selection_uses_kl=False,bridge_matrix_errors=errors)
    if (OUT/'plan.json').exists(): assert json.loads((OUT/'plan.json').read_text())==plan
    else: save_json(plan,OUT/'plan.json')
    records=[]
    for first in range(FIRST,FIRST+SEQUENCES):
        expected=[OUT/f'{first}_{family}_{budget}.json' for family in ['gim','vpd_exact'] for budget in ['native','200','512']]
        if all(p.exists() for p in expected):
            records.extend(json.loads(p.read_text()) for p in expected); continue
        rows=torch.tensor(np.array(data[first:first+1],dtype=np.int64),device='cuda'); ids=rows[:,:-1]
        (baseline,inputs,outputs,attention,checks),capture_seconds=timed(lambda:capture(model,ids))
        logp=baseline.log_softmax(-1)
        # Comparators only. These gradient caches are never passed to IFR/reader scoring.
        with torch.enable_grad():
            fc,fisher_seconds=timed(lambda:capture_fisher(model,rows,k=128,seed=6_100_000+first,microbatch=1))
            gc,gim_capture_seconds=timed(lambda:capture_gim(model,rows,microbatch=1))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            cached=trainer.model(ids,cache_type='input')
            ci=trainer.model.calc_causal_importances(cached.cache,sampling='continuous').lower_leaky
            native=torch.cat([ci[n].float()[0] for n in MODULES],-1)>0
            del cached,ci
        mapped=bridge_tools.remap(native,old,exact)
        counts=native.sum(-1)
        for family,pieces in [('gim',ours),('vpd_exact',exact)]:
            (positive,signed,hr),score_seconds=timed(lambda:local_scores(model,pieces,inputs))
            (fisher,_),rank_seconds=timed(lambda:audit.rank_and_census(fc,gc,pieces))
            _,gim_rank_seconds=timed(lambda:[(e['h'].float()@p.read).abs()*(e['events']['g'].float()@p.write).abs()
                                           for e,p in zip(gc,pieces,strict=True)])
            lp,ls=torch.cat(positive,-1),torch.cat(signed,-1)
            for budget in ['native','200','512']:
                dest=OUT/f'{first}_{family}_{budget}.json'
                if dest.exists(): records.append(json.loads(dest.read_text()));continue
                budget_counts=counts if budget=='native' else torch.full_like(counts,int(budget))
                choose=lambda scores:bridge_tools.variable_support(scores,budget_counts)
                masks=dict(fisher=choose(fisher),ifr_positive=choose(lp),ifr_signed=choose(ls))
                start=masks['ifr_signed']; route_history=[]
                for iteration in range(ROUNDS):
                    (sc,diagnostics),seconds=timed(lambda:condition(model,pieces,start,signed,hr,inputs,outputs,attention))
                    selected=choose(sc)
                    route_history.append(dict(round=iteration+1,seconds=seconds,changed_per_token=float((selected & ~start).sum(-1).float().mean()),**diagnostics))
                    masks[f'reader_{iteration+1}']=selected; start=selected
                if family=='vpd_exact' and budget=='native': masks['native_witness']=mapped
                record=dict(sequence=first,family=family,budget=budget,job_id=os.environ['SLURM_JOB_ID'],
                            timing=dict(capture=capture_seconds,local_two_scores=score_seconds,fisher_cache=fisher_seconds,
                                        gim_cache=gim_capture_seconds,gim_rank=gim_rank_seconds,fisher_and_gim_rank=rank_seconds),
                            capture_checks=checks,route_history=route_history,variants={})
                for label,keep in masks.items():
                    assert torch.equal(keep.sum(-1),budget_counts)
                    record['variants'][label]=dict(**audit.diagnose(model,ids,pieces,keep,logp,{n:h[None] for n,h in inputs.items()}),
                        count_mean=float(keep.sum(-1).float().mean()),
                        allocation=[float(m.sum(-1).float().mean()) for m in keep.split([p.count for p in pieces],-1)],
                        signed_score_mass=float((ls*keep).sum(-1).mean()/24),
                        positive_score_mass=float((lp*keep).sum(-1).mean()/24),
                        positive_zero_selected_fraction=float(((lp==0)&keep).sum()/keep.sum().clamp_min(1)))
                    if family=='vpd_exact':
                        record['variants'][label]['native_recall']=float((mapped&keep).sum()/mapped.sum().clamp_min(1))
                sp=OUT/f'{first}_{family}_{budget}_supports.npz'
                np.savez_compressed(sp,**{name:keep.nonzero().cpu().numpy().astype(np.uint16) for name,keep in masks.items()})
                record['supports_sha256']=sha(sp)
                save_json(record,dest); records.append(record)
                print(json.dumps(dict(sequence=first,family=family,budget=budget,kl={n:v['kl'] for n,v in record['variants'].items()},
                                      score_seconds=score_seconds,route_seconds=[r['seconds'] for r in route_history])),flush=True)
        del fc,gc,baseline,logp,inputs,outputs,attention
    summarize(records,plan)
    receipts=[]
    for p in OUT.iterdir():
        if p.is_file() and p.name not in ['complete.json','r2_receipts.json']:
            receipts.append(upload(p,'experiments/ifr67_20260915/'+p.name))
    for p in Path(__file__).parent.iterdir():
        if p.is_file(): receipts.append(upload(p,'experiments/ifr67_20260915/source/'+p.name))
    save_json(receipts,OUT/'r2_receipts.json')
    save_json(dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(OUT/'comparison.json')),OUT/'complete.json')
    upload(OUT/'complete.json','experiments/ifr67_20260915/complete.json')
    print('IFR67_COMPLETE',flush=True)


if __name__=='__main__': main()
