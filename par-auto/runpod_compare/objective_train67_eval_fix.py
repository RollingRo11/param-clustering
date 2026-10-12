"""Common matched-count KLKeep, plus clearly labelled secondary selectors."""
import json
import os
from pathlib import Path
import statistics as st
import math
import subprocess
import sys

ROOT=Path(os.environ['PAIR_ROOT'])
sys.path.insert(0,str(ROOT/'code/objective_train67_20260915_v3'))
sys.path.insert(0,str(ROOT/'code/first_nonlinearity67_20260915'))
import first_nonlinearity67 as first
from common import save_json,upload,credentials,sha

OUT=ROOT/'output/objective_train67_20260915_v3'
COUNTS='0,32,64,128,192,256,384,512,768,1024,2048,4096,8192,16384'


def native_finite(variant,checkpoint,out,start):
    """Bounded secondary curve; primary remains common Fisher on 128 contexts."""
    import numpy as np
    import torch
    from common import target,tokens
    from objective_train67 import capture_batch
    dest=out/'native_finite';dest.mkdir(exist_ok=True)
    if (dest/'klkeep.json').exists():return
    model=target();data,manifest=tokens('val')
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    pieces=[first.core.FreePieces(p['read'].cuda(),p['write'].cuda()) for p in saved['factors']]
    budgets=[int(k) for k in COUNTS.split(',')]+[sum(p.count for p in pieces)]
    records={k:[] for k in budgets}
    for seq in range(start,start+16):
        rows=torch.tensor(np.array(data[seq:seq+1],dtype=np.int64),device='cuda')
        ctx=capture_batch(model,rows,'gim' if variant=='finite_gim' else 'fisher',9200000+seq)[0]
        with torch.no_grad():
            scores=torch.cat([torch.cat([first.nonlinear_chunk(model,ctx,i,p.read[:,c:c+16],p.write[:,c:c+16]).abs()
                                        for c in range(0,p.count,16)],-1) for i,p in enumerate(pieces)],-1)
            order=scores.argsort(dim=-1,descending=True,stable=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):base=model(rows[:,:-1])[0]
            logp=base.float().log_softmax(-1)
            for k in budgets:
                keep=torch.zeros_like(scores).scatter_(-1,order[:,:k],1.)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=first.audit.run(model,rows[:,:-1],pieces,keep)
                records[k].append(float(first.audit.loss(pred,logp)))
        del ctx
    save_json(dict(sensor='Exact first-nonlinearity native scalar ranking',start_sequence=start,sequences=16,
                   note='Secondary curve on the first 16 contexts only; compare common curves on the same subset.',
                   target_sha256=first.TARGET_SHA,data_sha256=manifest['val']['sha256'],checkpoint_sha256=sha(checkpoint),
                   curve=[dict(kept_components=k,kl=st.mean(v),sequence_kl=v) for k,v in records.items()]),dest/'klkeep.json')


def report():
    baseline=OUT/'gim_baseline/eval_5000/fisher/klkeep.json'
    if not baseline.exists():return
    control=json.loads(baseline.read_text());allrows={}
    lines=['# 5k objective training: actual KLKeep','',
           'Common Fisher K64 selection, identical counts at each source position, 128 held-out contexts 512-639. '
           'All trials start from fresh seed-0 factors and process 163,840,000 positions. '
           'Training uses sampled component-position penalties; normal GIM uses its original dense penalty. '
           'Intervals are paired across contexts, not across training seeds.','']
    for variant in ['finite_gim','finite_fisher','write','pair_gim']:
        path=OUT/variant/'eval_5000/fisher/klkeep.json'
        if not path.exists():continue
        trial=json.loads(path.read_text())
        assert all(trial[k]==control[k] for k in ['target_sha256','data_sha256','training_data_sha256','start_sequence','positions','keep_counts','training_steps'])
        rows=[];lines += ['## '+variant,'','| Retained per token | Normal GIM KL | Trial KL | Difference | Paired 95% interval |','|---:|---:|---:|---:|---:|']
        for a,b in zip(control['curve'],trial['curve'],strict=True):
            assert a['kept_components']==b['kept_components']
            differences=[y-x for x,y in zip(a['sequence_kl'],b['sequence_kl'],strict=True)]
            d=st.mean(differences);h=1.98*st.stdev(differences)/math.sqrt(len(differences))
            rows.append(dict(kept=a['kept_components'],gim_kl=a['kl'],trial_kl=b['kl'],difference=d,interval=[d-h,d+h]))
            lines.append(f"| {int(a['kept_components'])} | {a['kl']:.5f} | {b['kl']:.5f} | {d:.5f} | {d-h:.5f}, {d+h:.5f} |")
        allrows[variant]=dict(rows=rows,smallest_tested_count_at_kl={str(t):min((int(r['kept_components']) for r in trial['curve'] if r['kl']<=t),default=None) for t in [.1,.3,1.]})
    save_json(allrows,OUT/'comparison.json');(OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    assert os.environ.get('SLURM_JOB_ID');credentials()
    variant,step=sys.argv[1],int(sys.argv[2]);out=OUT/variant/f'eval_{step}';out.mkdir(parents=True,exist_ok=True)
    if variant=='gim_baseline':checkpoint=ROOT/'output/gim_1h/checkpoints/005000/model.pth';method='parauto';expected=5000
    elif variant=='vpd_published':checkpoint=ROOT/'checkpoints/published_vs_finished/vpd.pth';method='vpd';expected=400000
    else:checkpoint=OUT/variant/'checkpoints'/f'{step:06d}'/'model.pth';method='parauto';expected=step
    count=128 if step==5000 else 16;start=512 if step==5000 else 480
    sensors=['gradient'] if method=='vpd' else ['fisher','gim']+(['write'] if variant=='write' else [])
    for sensor in sensors:
        dest=out/sensor
        if (dest/'klkeep.json').exists():continue
        cmd=[sys.executable,'-u',str(ROOT/'code/gim_count67_20260914/compare/gim_count67_evaluate.py'),method,
             '--checkpoint',str(checkpoint),'--expected-step',str(expected),'--output',str(dest),
             '--sensor',sensor,'--fisher-k','64','--sequences',str(count),'--start-sequence',str(start),
             '--keep-counts',COUNTS,'--run-tag',f'{variant}-{step}-{sensor}-objective-train67']
        subprocess.run(cmd,check=True)
    if variant in ['finite_gim','finite_fisher']:native_finite(variant,checkpoint,out,start)
    receipts=[]
    for p in sorted(out.rglob('*.json')):
        if 'wandb' not in p.parts:receipts.append(upload(p,f'experiments/objective_train67_20260915_v3/{p.relative_to(OUT)}'))
    save_json(dict(complete=True,verified_uploads=len(receipts)),out/'complete.json');report()
    if (OUT/'report.md').exists():
        upload(OUT/'report.md','experiments/objective_train67_20260915_v3/report.md')
        upload(OUT/'comparison.json','experiments/objective_train67_20260915_v3/comparison.json')
    print('EVALUATION_COMPLETE',variant,step,flush=True)


if __name__=='__main__':main()
