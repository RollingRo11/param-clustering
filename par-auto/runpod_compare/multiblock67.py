"""Contiguous multi-block sufficiency diagnostics on the frozen 67M target.

Compare all 10 spans, lengths 1–4, with live error propagation within each span.
Fixed supports, no factor updates; full and partial endpoint KL are controls.
"""
import json
import os
from pathlib import Path
import statistics as st
import sys

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/objective67_20260915'))
import objective67 as obj
audit, bridge, core = obj.audit, obj.bridge, obj.core
import numpy as np
import torch
from common import MODULES, TARGET_SHA, target, tokens, save_json, sha, upload, credentials
from fisher_attribution import capture_fisher
from gim_attribution import capture_gim

EXPERIMENT = 'multiblock67_20260915_v2'
OUT = ROOT/'output'/EXPERIMENT
REPRODUCTION = list(range(448,456))
CONFIRMATION = list(range(456,464))


def boundary(model, ctx, end):
    name = f'h.{end}.rms_1' if end < 4 else 'ln_f'
    return ctx['inputs'][name], model.get_submodule(name)


def span_forward(model, ctx, pieces, keep, start, end):
    x = ctx['inputs'][f'h.{start}.rms_1'][None]
    with audit.intervention(model, pieces, audit.masks_of(keep, pieces)):
        for block in model.h[start:end]:
            x = block(x)
    return x[0]


@torch.no_grad()
def span_suite(model, ctx, pieces, keep):
    result = {}
    logp = ctx['base'].log_softmax(-1)
    parts = keep.split([p.count for p in pieces], -1)
    for start in range(4):
        x = ctx['inputs'][f'h.{start}.rms_1'][None]
        states = []
        with audit.intervention(model, pieces, audit.masks_of(keep, pieces)):
            for end in range(start+1, 5):
                x = model.h[end-1](x)
                states.append((end, x))
        for end, x in states:
            ref, norm = boundary(model, ctx, end)
            pred = x[0]
            entry = dict(start=start, end=end, length=end-start,
                         raw_mse=float(obj.rel_mse(pred, ref)), rms_mse=float(obj.rel_mse(norm(pred), norm(ref))),
                         cosine=float(obj.cosine(pred, ref)),
                         count_in_span=float(torch.cat(parts[start*6:end*6], -1).sum(-1).float().mean()))
            # Restore the intact prefix/suffix around this selected span.
            # Unchanged surrounding computation is not charged: diagnostic only.
            suffix = x
            for block in model.h[end:]:
                suffix = block(suffix)
            logits = model.lm_head(model.ln_f(suffix))
            entry['hybrid_kl'] = float(audit.loss(logits, logp))
            result[f'{start}:{end}'] = entry
    means = {str(length):{metric:st.mean(row[metric] for row in result.values() if row['length']==length)
                         for metric in ['raw_mse','rms_mse','cosine']}
             for length in range(1,5)}
    return dict(spans=result, mean_by_length=means, full_kl=result['0:4']['hybrid_kl'])


@torch.no_grad()
def validate(model, ctx, pieces, keep):
    all_kept = span_suite(model, ctx, pieces, torch.ones_like(keep))
    assert max(v['raw_mse'] for v in all_kept['spans'].values()) < 1e-8
    measured = span_suite(model, ctx, pieces, keep)
    old = obj.conditional_losses(model, ctx, pieces, keep)
    for a,b in [('rms_mse','block_rms'),('raw_mse','block_mse'),('cosine','block_cosine')]:
        assert abs(measured['mean_by_length']['1'][a]-float(old[b])) < 2e-5
    # Check every partial-span suffix against a complete model intervention.
    errors = {}
    for start in range(4):
        for end in range(start+1,5):
            expected = float(audit.loss(audit.run(model, ctx['rows'][:, :-1], pieces, keep,
                                                 layers=tuple(range(start,end))), ctx['base'].log_softmax(-1)))
            observed = measured['spans'][f'{start}:{end}']['hybrid_kl']
            errors[f'{start}:{end}'] = abs(expected-observed)
            assert abs(expected-observed) < 2e-5, (start,end,expected,observed)
    return dict(all_kept_max_mse=max(v['raw_mse'] for v in all_kept['spans'].values()),
                single_block_matches_previous=True, hybrid_kl_max_errors=errors,
                full_kl=measured['full_kl'])


def benchmark(model, ctx, pieces, keep):
    # A backward-cost diagnostic only: no optimizer and no parameter mutation.
    p = obj.clone(pieces, grad=True)
    params = [t for x in p for t in [x.read,x.write]]
    result = {}
    obj.trace(model, ctx['rows'], sensors=False)
    trace_seconds = st.median(obj.timed(lambda: obj.trace(model, ctx['rows'], sensors=False))[1] for _ in range(3))
    capture_gim(model, ctx['rows'], microbatch=1)
    captures = [obj.timed(lambda: capture_gim(model, ctx['rows'], microbatch=1)) for _ in range(3)]
    cache, gim_cache_seconds = captures[-1][0], st.median(t for _,t in captures)
    base_ctx = dict(ctx, cache={'gim':cache})
    scales = json.loads((ROOT/'output/objective67_20260915/normalizers.json').read_text())
    def base_run():
        loss = obj.reduce_scores(obj.scores(model, base_ctx, p, 'gim'), 'gim', .641, scales)
        torch.autograd.grad(loss, params)
        return float(loss.detach())
    base_run()
    baseline = [obj.timed(base_run)[1] for _ in range(3)]
    result['gim'] = dict(cache_seconds=gim_cache_seconds, forward_backward_seconds=st.median(baseline),
                         total_proxy=gim_cache_seconds+st.median(baseline))
    for length in range(1,5):
        def run():
            losses = []
            for start in range(5-length):
                end = start+length
                pred = span_forward(model, ctx, p, keep, start, end)
                ref,norm = boundary(model, ctx, end)
                losses.append(obj.rel_mse(norm(pred),norm(ref)))
            loss = torch.stack(losses).mean()
            torch.autograd.grad(loss, params)
            return float(loss.detach())
        run()
        times = [obj.timed(run)[1] for _ in range(3)]
        proxy = trace_seconds+st.median(times)
        result[str(length)] = dict(cache_seconds=trace_seconds,forward_backward_seconds=st.median(times),
                                   total_proxy=proxy, relative_to_gim=proxy/result['gim']['total_proxy'],
                                   block_executions=length*(5-length))
    return result


def finish(plan, records, timing):
    summaries = {}
    for split in ['reproduction','confirmation']:
        rows = [r for r in records if r['split']==split]
        values = {}
        for length in range(1,5):
            k = str(length)
            get = lambda r,state,selector,metric: r['states'][state][selector]['mean_by_length'][k][metric]
            metrics = {}
            for metric in ['rms_mse','raw_mse','cosine']:
                v = [get(r,'vpd_exact','inherited_native',metric) for r in rows]
                g = [get(r,'gim5k','fisher',metric) for r in rows]
                metrics[metric] = dict(vpd=st.mean(v),gim=st.mean(g),ratio=st.mean(v)/st.mean(g),
                    vpd_preferred_fraction=st.mean(float(a<b) for a,b in zip(v,g)),
                    native_beats_fisher_fraction=st.mean(float(get(r,'vpd_exact','inherited_native',metric)<get(r,'vpd_exact','fisher',metric)) for r in rows),
                    native_beats_random_fraction=st.mean(float(get(r,'vpd_exact','inherited_native',metric)<get(r,'vpd_exact','random_native_allocation',metric)) for r in rows),
                    worsening_trajectory_increases_loss_fraction=st.mean(sum(get(r,b,'inherited_native',metric)>get(r,a,'inherited_native',metric)
                        for a,b in zip(['vpd_exact','descent_20','descent_100'],['descent_20','descent_100','descent_200']))/3 for r in rows))
            values[k] = metrics
        summaries[split] = values
    endpoints = {split:{name:st.mean(r['states'][name][selector]['full_kl'] for r in records if r['split']==split)
                       for name,selector in [('vpd_exact','inherited_native'),('gim5k','fisher')]}
                 for split in ['reproduction','confirmation']}
    result = dict(plan=plan,summaries=summaries,endpoints=endpoints,timing=timing,records=records)
    save_json(result,OUT/'comparison.json')
    lines = ['# Multi-block error screening','',
             'Frozen 67M target on Explorer; no factor updates. Same supports and counts as the objective screen. '
             'Eight previous contexts check reproducibility; eight fresh contexts confirm the predeclared span-length comparison. '
             'Cross-run differences are recorded separately from strict same-run implementation checks.', '',
             '| Split | Span length | VPD normalized error | GIM normalized error | VPD/GIM | VPD preferred | Native beats Fisher | Damage raises loss |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, table in summaries.items():
        for length, entry in table.items():
            v=entry['rms_mse']
            lines.append(f"| {split} | {length} | {v['vpd']:.5f} | {v['gim']:.5f} | {v['ratio']:.4f} | {v['vpd_preferred_fraction']:.2f} | {v['native_beats_fisher_fraction']:.2f} | {v['worsening_trajectory_increases_loss_fraction']:.2f} |")
    lines += ['', 'Counts are matched globally at every source position. Errors average all contiguous spans of the indicated length. '
              'Every span starts once at its intact input and propagates the selected state through all blocks inside it. '
              'Full-model KL on fresh contexts: VPD **'+f"{endpoints['confirmation']['vpd_exact']:.5f}"+'**, GIM **'+f"{endpoints['confirmation']['gim5k']:.5f}"+'**.', '',
              '## Individual span locations on fresh contexts','',
              '| Blocks (zero-based, inclusive) | VPD normalized error | GIM normalized error | VPD/GIM | VPD hybrid KL | GIM hybrid KL |',
              '|---|---:|---:|---:|---:|---:|']
    reproduction = [r['reproduction_checks'] for r in records if r['split']=='reproduction']
    lines[4:4] = ['Maximum absolute cross-run KL difference: '+str(max(abs(v['kl_delta']) for r in reproduction for v in r.values()))+'. '
                  'The previous screen did not save its support indices; regenerated supports are not assumed bitwise identical. '
                  'Strict same-run comparisons with the original single-block implementation and complete model forwards are required.', '']
    rows=[r for r in records if r['split']=='confirmation']
    for start in range(4):
        for end in range(start+1,5):
            def mean(state,selector,metric):return st.mean(r['states'][state][selector]['spans'][f'{start}:{end}'][metric] for r in rows)
            v,g=mean('vpd_exact','inherited_native','rms_mse'),mean('gim5k','fisher','rms_mse')
            vk,gk=mean('vpd_exact','inherited_native','hybrid_kl'),mean('gim5k','fisher','hybrid_kl')
            lines.append(f"| {start}–{end-1} | {v:.5f} | {g:.5f} | {v/g:.4f} | {vk:.5f} | {gk:.5f} |")
    lines += ['', 'Hybrid KL keeps the surrounding prefix/suffix intact. It is a mechanistic diagnostic with uncharged surrounding computation, '
              'not a fair lower-L0 comparison. Only the four-block endpoint executes the whole selected model.', '',
              '## Cost diagnostic','',
              '| Mean span objective | Block executions | Capture + forward/backward seconds | Relative to GIM |',
              '|---|---:|---:|---:|']
    for length in range(1,5):
        row=timing[str(length)]
        lines.append(f"| {length} | {row['block_executions']} | {row['total_proxy']:.5f} | {row['relative_to_gim']:.3f} |")
    lines += ['', 'Single-context FP32 factor arithmetic, median of three warmed measurements. GIM uses the established BF16 derivative cache. '
              'Forward-only trace capture includes validation overhead. All objectives exclude optimizer and exact-sum projection from timing. '
              'The prototype hooks also compute an unused intact linear output before replacement. These are diagnostic timings, not production-batch training benchmarks.', '',
              '## Interpretation limits','',
              'All span objectives receive a fixed binary support. The four-block version is a full selected-model hidden-state reconstruction objective. '
              'Acceptance does not make it mask-free, provide a cheap selector, or prove a lower L0 floor. '
              'Gradients were computed for timing only; no factors were updated. Raw and cosine metrics and all per-context outcomes are in comparison.json.', '']
    (OUT/'report.md').write_text('\n'.join(lines))
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed();OUT.mkdir(parents=True,exist_ok=True)
    model=target();data,manifest=tokens('val')
    ours,_,oursmeta=audit.shared.load(model,'gim_5000')
    old,trainer,oldmeta=audit.shared.load(model,'vpd_published')
    exact=obj.read_pieces(ROOT/'output/gap67_bridge_20260915/bridge_model.pth')
    states=dict(vpd_exact=exact,gim5k=ours)
    for step in [20,100,200]:states[f'descent_{step}']=obj.read_pieces(ROOT/f'output/gap67_descent_20260915/{step}.pth')
    plan=dict(source_sha256=sha(__file__),target_sha256=TARGET_SHA,val_sha256=manifest['val']['sha256'],
              sources=dict(gim=oursmeta,vpd=oldmeta),reproduction=REPRODUCTION,confirmation=CONFIRMATION,
              bridge_sha256=sha(ROOT/'output/gap67_bridge_20260915/bridge_model.pth'),
              trajectory_sha256={str(s):sha(ROOT/f'output/gap67_descent_20260915/{s}.pth') for s in [20,100,200]},
              spans=[(s,e) for s in range(4) for e in range(s+1,5)],
              note='Fixed-support diagnostics only; no factor updates. Span length four runs the full selected transformer.',
              job_id=os.environ['SLURM_JOB_ID'])
    save_json(plan,OUT/'plan.json');records=[];timing=None
    for sequence in REPRODUCTION+CONFIRMATION:
        ctx=obj.trace(model,torch.tensor(np.array(data[sequence:sequence+1],dtype=np.int64),device='cuda'),sensors=False)
        native=obj.native_support(trainer,ctx)>0
        counts=native.sum(-1)
        inherited=bridge.remap(native,old,exact)
        if sequence==REPRODUCTION[0]:
            save_json(validate(model,ctx,exact,inherited.float()),OUT/'validation.json')
            repeat = obj.native_support(trainer,ctx)>0
            old_style_ctx = obj.trace(model,ctx['rows'],sensors=True)
            after_trace = obj.native_support(trainer,old_style_ctx)>0
            numerical = dict(native_repeat_disagreements=int((native!=repeat).sum()),
                             native_after_sensor_capture_disagreements=int((native!=after_trace).sum()),
                             clean_logits_max_difference=float((ctx['base']-old_style_ctx['base']).abs().max()),
                             native_count_mean=float(counts.float().mean()))
            save_json(numerical,OUT/'numerical_reproduction.json')
            print('NUMERICAL_REPRODUCTION '+json.dumps(numerical),flush=True)
            assert numerical['native_repeat_disagreements']==0
            del old_style_ctx,repeat,after_trace
            print('MULTIBLOCK67_PREFLIGHT_PASSED',flush=True)
            timing=benchmark(model,ctx,exact,inherited.float());save_json(timing,OUT/'timing.json')
        fisher=capture_fisher(model,ctx['rows'],k=64,seed=6419000+sequence,microbatch=1)
        record=dict(sequence=sequence,split='reproduction' if sequence in REPRODUCTION else 'confirmation',
                    count_mean=float(counts.float().mean()),states={})
        with torch.no_grad():
            for name,pieces in states.items():
                fs=torch.cat([(e['h'].float()@p.read).abs()*(e['events']['cot'].float()@p.write).square().mean(1).sqrt()
                              for e,p in zip(fisher,pieces,strict=True)],-1)
                variants=dict(fisher=bridge.variable_support(fs,counts))
                if name!='gim5k':variants['inherited_native']=inherited
                generator=torch.Generator(device='cuda').manual_seed(73000+sequence)
                random=torch.rand(inherited.shape,generator=generator,device='cuda')
                variants['random_native_allocation']=torch.cat([bridge.variable_support(s,m.sum(-1)) for s,m in
                    zip(random.split([p.count for p in pieces],-1),inherited.split([p.count for p in pieces],-1),strict=True)],-1)
                record['states'][name]={}
                supports={}
                for label,mask in variants.items():
                    assert torch.equal(mask.sum(-1),counts)
                    record['states'][name][label]=span_suite(model,ctx,pieces,mask.float())
                    supports[label]=mask.nonzero().cpu().numpy().astype(np.uint16)
                np.savez_compressed(OUT/f'{sequence}_{name}_supports.npz',**supports)
        if sequence in REPRODUCTION:
            prior=json.loads((ROOT/f'output/objective67_20260915/{sequence}.json').read_text())
            record['reproduction_checks']={}
            for name,label in [('vpd_exact','inherited_native'),('gim5k','fisher')]:
                expected=prior['states'][name]['variants'][label]
                observed=record['states'][name][label]
                record['reproduction_checks'][name]=dict(kl_delta=observed['full_kl']-expected['kl'],
                    block_rms_delta=observed['mean_by_length']['1']['rms_mse']-expected['conditional']['block_rms'],
                    count_delta=record['count_mean']-prior['count_mean'])
                idx=np.load(OUT/f'{sequence}_{name}_supports.npz')[label].astype(np.int64)
                mask=torch.zeros_like(inherited,dtype=torch.float32)
                mask[idx[:,0],idx[:,1]]=1
                with torch.no_grad():
                    same_kl=obj.endpoint(model,ctx,states[name],mask)
                    same_rms=float(obj.conditional_losses(model,ctx,states[name],mask)['block_rms'])
                assert abs(same_kl-observed['full_kl'])<2e-5
                assert abs(same_rms-observed['mean_by_length']['1']['rms_mse'])<2e-5
            print('CROSS_RUN_REPRODUCTION '+json.dumps(record['reproduction_checks']),flush=True)
        save_json(record,OUT/f'{sequence}.json');records.append(record)
        print(json.dumps(dict(sequence=sequence,ratios={str(k):record['states']['vpd_exact']['inherited_native']['mean_by_length'][str(k)]['rms_mse']/
                    record['states']['gim5k']['fisher']['mean_by_length'][str(k)]['rms_mse'] for k in range(1,5)})),flush=True)
        del ctx,fisher
    finish(plan,records,timing)
    receipts=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in ['complete.json','analysis_complete.json','r2_receipts.json']:
            receipts.append(upload(p,f'experiments/{EXPERIMENT}/{p.name}'))
    for p in sorted(Path(__file__).parent.iterdir()):
        if p.is_file():receipts.append(upload(p,f'experiments/{EXPERIMENT}/source/{p.name}'))
    save_json(receipts,OUT/'r2_receipts.json')
    completed=dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(OUT/'comparison.json'))
    save_json(completed,OUT/'complete.json');upload(OUT/'complete.json',f'experiments/{EXPERIMENT}/complete.json')
    save_json(completed,OUT/'analysis_complete.json')
    print('MULTIBLOCK67_COMPLETE',flush=True)


if __name__=='__main__':main()
