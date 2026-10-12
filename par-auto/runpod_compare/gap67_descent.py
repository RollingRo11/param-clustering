"""Does unmasked GIM training preserve the converted VPD sparse solution?"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import statistics as st

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/gap67_bridge_20260915'))
import gap67_bridge as bridge_tools
audit = bridge_tools.audit
import torch
import numpy as np
from common import target, tokens, batch, save_json, sha, upload, credentials, MODULES, TARGET_SHA
from train_pair import core, method
from gim_attribution import capture_gim
from fisher_attribution import capture_fisher

OUT = ROOT/'output/gap67_descent_20260915'
BRIDGE = ROOT/'output/gap67_bridge_20260915/bridge_model.pth'


def read_pieces(path):
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['target_sha256'] == TARGET_SHA
    return [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]


def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    OUT.mkdir(parents=True, exist_ok=True)
    model = target()
    training, manifest = tokens('train')
    control = ROOT/'output/gim_1h/checkpoints/005000/checkpoint.pt'
    saved = torch.load(control, map_location='cpu', mmap=True, weights_only=False)
    cfg = method.TrainConfig(**saved['state']['config'])
    cfg.log_interval = 20
    assert cfg.steps == 8700 and cfg.count_coeff == 0
    refs = [s.cuda() for s in saved['state']['initial_scores']]
    initial = read_pieces(BRIDGE)
    # Deliberately construct a fresh-Adam start at the original schedule's
    # 5,000-step point. No prior optimizer moments belong to these factors.
    resume = dict(config=asdict(cfg), next_step=5000,
                  pieces=[dict(read=p.read, write=p.write) for p in initial],
                  optimizer=dict(state={}, param_groups=saved['state']['optimizer']['param_groups']),
                  batches=saved['state']['batches'],
                  scaler=saved['state']['scaler'], initial_scores=refs, histories=[[] for _ in MODULES],
                  overflow_skips=0)
    del saved
    plan = dict(source_sha256=sha(__file__), bridge_sha256=sha(BRIDGE), target_sha256=TARGET_SHA,
                train_sha256=manifest['train']['sha256'], val_sha256=manifest['val']['sha256'],
                updates=200, schedule_start=5000, configuration=asdict(cfg),
                checkpoints=[0,20,100,200], first_validation_sequence=424, validation_sequences=8,
                note='Diagnostic warm start from VPD-converted factors, not an independently learned method. '
                     'Fresh Adam, same fixed GIM normalizers; original schedule window 5000–5199; '
                     'unmodified-target GIM caches only during training. Training batches 0–199; no masks or CI used for training.')
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text()) == json.loads(json.dumps(plan))
    else:
        save_json(plan, OUT/'plan.json')
    if not (OUT/'200.pth').exists():
        first = capture_gim(model, batch(training, 0))
        reference = [method.slice_entry(e, slice(0,256)) for e in first]
        pending = [first]
        def provider(step):
            return pending.pop() if step == 5000 and pending else capture_gim(model, batch(training, step-5000))
        def checkpoint(state):
            update = state['next_step']-5000
            if update in [20,100,200]:
                torch.save(dict(target_sha256=TARGET_SHA, updates=update,
                                factors=[{k:v.detach().cpu() for k,v in p.items()} for p in state['pieces']]), OUT/f'{update}.pth')
        logs=[]
        def log(row):
            if row['matrix_index'] == 23:
                record=dict(update=row['step']+1-5000, total_loss=row['total_loss'], p=row['p'])
                logs.append(record); save_json(logs, OUT/'training.json'); print(json.dumps(record), flush=True)
        trained, _ = method.fit(reference, [model.get_submodule(n).weight for n in MODULES], initial,
                                cfg, resume=resume, stop_after=200, batch_provider=provider,
                                checkpoint=checkpoint, checkpoint_interval=20, log=log)
        del trained, reference, first, pending
    del training, resume
    torch.cuda.empty_cache()
    validation, _ = tokens('val')
    original, trainer, _ = audit.shared.load(model, 'vpd_published')
    candidates={0:initial, **{step:read_pieces(OUT/f'{step}.pth') for step in [20,100,200]}}
    result = dict(plan=plan, stages={str(step):[] for step in candidates})
    for first in range(424,432):
        rows=torch.tensor(np.array(validation[first:first+1],dtype=np.int64),device='cuda'); ids=rows[:,:-1]
        base, inputs=audit.clean_capture(model,ids); logp=base.log_softmax(-1)
        gc=capture_gim(model,rows,microbatch=1)
        fc=capture_fisher(model,rows,k=128,seed=6_100_000+first,microbatch=1)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            cached=trainer.model(ids,cache_type='input')
            ci=trainer.model.calc_causal_importances(cached.cache,sampling='continuous').lower_leaky
            positive=torch.cat([ci[n].float()[0]>0 for n in MODULES],-1)
            del cached,ci
        for step,pieces in candidates.items():
            entry=dict(first_sequence=first, objective=bridge_tools.objective(gc,pieces,refs),
                       matrix_sum_errors=[core.reconstruction_error(model.get_submodule(n).weight,p) for n,p in zip(MODULES,pieces)])
            assert max(entry['matrix_sum_errors']) < 1e-5
            inherited=bridge_tools.remap(positive,original,pieces)
            entry['native']=audit.diagnose(model,ids,pieces,inherited,logp,inputs)
            scores,_=audit.rank_and_census(fc,gc,pieces)
            ranked=bridge_tools.variable_support(scores,positive.sum(-1))
            entry['fisher']=audit.diagnose(model,ids,pieces,ranked,logp,inputs)
            best,history=audit.endpoint_search(model,ids,pieces,logp,inherited,32)
            entry['native_refined']=dict(**audit.diagnose(model,ids,pieces,best,logp,inputs),search=history)
            result['stages'][str(step)].append(entry)
            save_json(result,OUT/'progress.json')
        print(json.dumps(dict(evaluated_sequence=first)),flush=True)
    lines=['# Unmasked GIM descent from the exact VPD bridge','',plan['note'],'',
           '| Updates | Common GIM Lp p=.641 | Soft count | Inherited native KL | Refined native KL | Fisher KL at matched native budget |',
           '|---:|---:|---:|---:|---:|---:|']
    result['summary']={}
    for step,rows in result['stages'].items():
        s=dict(lp=st.mean(r['objective']['0.641'] for r in rows),soft_count=st.mean(r['objective']['soft_count'] for r in rows),
               native=st.mean(r['native']['kl'] for r in rows),refined=st.mean(r['native_refined']['kl'] for r in rows),
               fisher=st.mean(r['fisher']['kl'] for r in rows))
        result['summary'][step]=s
        lines.append(f"| {step} | {s['lp']:.5f} | {s['soft_count']:.1f} | {s['native']:.5f} | {s['refined']:.5f} | {s['fisher']:.5f} |")
    lines+=['','Supports use approximately 193 components/token, with per-token native CI budgets held fixed. '
            'The native selector is frozen; refinement is bounded binary endpoint search. '
            'Failure of those selectors does not prove that all good supports were destroyed.','']
    save_json(result,OUT/'comparison.json');(OUT/'report.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
    receipts=[]
    for p in OUT.iterdir():
        if p.is_file() and p.name not in ['complete.json','r2_receipts.json']:
            receipts.append(upload(p,'experiments/gap67_descent_20260915/'+p.name))
    for p in Path(__file__).parent.iterdir():
        if p.is_file():receipts.append(upload(p,'experiments/gap67_descent_20260915/source/'+p.name))
    save_json(receipts,OUT/'r2_receipts.json')
    save_json(dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(OUT/'comparison.json')),OUT/'complete.json')
    upload(OUT/'complete.json','experiments/gap67_descent_20260915/complete.json')
    print('GAP67_DESCENT_COMPLETE',flush=True)


if __name__=='__main__':main()
