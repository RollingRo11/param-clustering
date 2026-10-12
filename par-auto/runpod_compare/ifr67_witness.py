"""Follow-up: does reader conditioning preserve a supplied good support?

Declared after the first IFR diagnostic, on the same inputs (not a new held-out
benchmark). Freeze factors; seed with native support; control allocation and
unchanged Q/K/final-Down supports. No optimization using KL.
"""
import os
from pathlib import Path
import sys
ROOT=Path(os.environ['PAIR_ROOT'])
sys.path.insert(0,str(ROOT/'code/ifr67_20260915'))
import ifr67 as base
from ifr67 import torch,np,audit,bridge_tools,core,MODULES,TARGET_SHA,target,tokens,save_json,sha,credentials,upload,timed
import json
import statistics as st
OUT=ROOT/'output/ifr67_witness_20260915'


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed();OUT.mkdir(parents=True,exist_ok=True)
    model=target();data,manifest=tokens('val')
    old,trainer,meta=audit.shared.load(model,'vpd_published')
    path=ROOT/'output/gap67_bridge_20260915/bridge_model.pth'
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    assert saved['target_sha256']==TARGET_SHA
    pieces=[core.FreePieces(p['read'].cuda(),p['write'].cuda()) for p in saved['factors']]
    del saved
    plan=dict(source_sha256=sha(__file__),base_sha256=sha(base.__file__),first=432,sequences=8,
              kind='post-result diagnostic on original inputs; no training or KL-based selection',
              readers=base.READERS,destinations=base.DESTINATIONS,bridge_sha256=sha(path),val_sha256=manifest['val']['sha256'])
    save_json(plan,OUT/'plan.json');records=[]
    for first in range(432,440):
        dest=OUT/f'{first}.json'
        if dest.exists():records.append(json.loads(dest.read_text()));continue
        rows=torch.tensor(np.array(data[first:first+1],dtype=np.int64),device='cuda');ids=rows[:,:-1]
        logits,inputs,outputs,attention,checks=base.capture(model,ids);logp=logits.log_softmax(-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            cached=trainer.model(ids,cache_type='input')
            ci=trainer.model.calc_causal_importances(cached.cache,sampling='continuous').lower_leaky
            original=torch.cat([ci[n].float()[0] for n in MODULES],-1)>0
            del cached,ci
        native=bridge_tools.remap(original,old,pieces)
        parts=native.split([p.count for p in pieces],-1)
        positive,signed,hr=base.local_scores(model,pieces,inputs)
        allocate=lambda scores:audit.allocated_support(scores,pieces,parts)
        masks=dict(native=native,positive_native_allocation=allocate(torch.cat(positive,-1)),
                   signed_native_allocation=allocate(torch.cat(signed,-1)))
        (scores,diagnostics),seconds=timed(lambda:base.condition(model,pieces,native,signed,hr,inputs,outputs,attention))
        masks['reader_native_seed_global']=bridge_tools.variable_support(scores,native.sum(-1))
        masks['reader_native_seed_allocation']=allocate(scores)
        chosen=masks['reader_native_seed_allocation'].split([p.count for p in pieces],-1)
        masks['reader_routed_modules_only']=torch.cat([q if (i%6 in [2,3,4] or (i%6==5 and i//6<3)) else n
                                                     for i,(q,n) in enumerate(zip(chosen,parts))],-1)
        record=dict(sequence=first,seconds=seconds,route_diagnostics=diagnostics,variants={})
        for label,keep in masks.items():
            assert torch.equal(keep.sum(-1),native.sum(-1))
            record['variants'][label]=dict(**audit.diagnose(model,ids,pieces,keep,logp,{n:h[None] for n,h in inputs.items()}),
                native_recall=float((native&keep).sum()/native.sum()),
                changed_per_token=float((keep&~native).sum(-1).float().mean()),
                routed_score_mass=float((scores*keep).sum(-1).mean()/24))
        sp=OUT/f'{first}_supports.npz'
        np.savez_compressed(sp,**{k:v.nonzero().cpu().numpy().astype(np.uint16) for k,v in masks.items()})
        record['supports_sha256']=sha(sp);save_json(record,dest);records.append(record)
        print(json.dumps(dict(sequence=first,kl={k:v['kl'] for k,v in record['variants'].items()})),flush=True)
    summary={}
    lines=['# IFR reader score: supplied-support and allocation controls','','Same contexts 432–439, converted VPD factors, native per-token budgets. No training.','',
           '| Selector | KL | Native recall | Changed components/token |', '|---|---:|---:|---:|']
    for label in records[0]['variants']:
        summary[label]={k:st.mean(r['variants'][label][k] for r in records) for k in ['kl','native_recall','changed_per_token','restore_last_kl','routed_score_mass']}
        v=summary[label];lines.append(f"| {label} | {v['kl']:.5f} | {v['native_recall']:.3f} | {v['changed_per_token']:.2f} |")
    lines+=['','Routed-modules-only preserves native Q/K and final Down supports, native counts in every matrix, and changes only matrices with implemented outgoing reader routes. '
            'These are score-driven replacements, not KL-accepted swaps. A deterioration measures alignment of this bounded scoring rule; it does not rule out other route objectives.','']
    save_json(dict(plan=plan,summary=summary,records=records),OUT/'comparison.json');(OUT/'report.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
    receipts=[]
    for p in OUT.iterdir():
        if p.is_file() and p.name not in ['complete.json','r2_receipts.json']:receipts.append(upload(p,'experiments/ifr67_witness_20260915/'+p.name))
    for p in Path(__file__).parent.iterdir():
        if p.is_file():receipts.append(upload(p,'experiments/ifr67_witness_20260915/source/'+p.name))
    save_json(receipts,OUT/'r2_receipts.json');save_json(dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(OUT/'comparison.json')),OUT/'complete.json')
    upload(OUT/'complete.json','experiments/ifr67_witness_20260915/complete.json');print('IFR67_WITNESS_COMPLETE',flush=True)


if __name__=='__main__':main()
