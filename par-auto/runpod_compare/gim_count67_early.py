"""Early KLKeep read, including step-matched 1k GIM and count checkpoints."""
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
ROOT=Path(os.environ['PAIR_ROOT'])
SNAP=ROOT/'code/gim_count67_20260914'
sys.path.insert(0,str(SNAP/'compare'))
from common import save_json, upload, sha, credentials

SPECS=[('gim_1000',1000,ROOT/'output/gim_1h/checkpoints/001000/model.pth'),
       ('count_1000',1000,ROOT/'output/gim_count67_20260914/checkpoints/001000/model.pth'),
       ('count_1382',1382,ROOT/'output/gim_count67_20260914/checkpoints/001382/model.pth'),
       ('gim_2000',2000,ROOT/'output/gim_1h/checkpoints/002000/model.pth'),
       ('gim_5000',5000,ROOT/'output/gim_1h/checkpoints/005000/model.pth')]


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Run on Explorer GPU allocation'
    credentials();out=ROOT/'output/gim_count67_20260914/early_1382';out.mkdir(parents=True,exist_ok=True)
    save_json(dict(job_id=os.environ['SLURM_JOB_ID'],source_sha256=sha(__file__),sequences=64,start_sequence=256,
                   ranking='shared Fisher K64',training_steps={n:s for n,s,_ in SPECS}),out/'plan.json')
    records={}
    for name,step,path in SPECS:
        subprocess.run([sys.executable,'-u',str(SNAP/'compare/gim_count67_evaluate.py'),'parauto',
                        '--expected-step',str(step),'--checkpoint',str(path),'--output',str(out/name),
                        '--sensor','fisher','--fisher-k','64','--sequences','64','--start-sequence','256',
                        '--run-tag','gim-count67-early-'+name],check=True)
        records[name]=json.loads((out/name/'klkeep.json').read_text())
        print('EARLY_CHECKPOINT_EVALUATED '+name,flush=True)
    assert len({(r['target_sha256'],r['data_sha256'],r['training_data_sha256'],r['positions'],r['start_sequence'],r['sensor']) for r in records.values()})==1
    curves={n:{int(x['kept_components']):x for x in r['curve']} for n,r in records.items()}
    lines=['# Early GIM + count KLKeep','',
           'Explorer H200; frozen 67M target; 64 shared validation contexts (256–319), 32,768 positions. '
           'All rankings use identical Fisher K64 seeds on the intact model. Counts are retained rank-one pieces per token.','',
           'The 1,000-step comparison is step matched. The 1,382-step trial versus 2,000/5,000-step GIM comparisons have different training budgets. '
           'These are early single-seed results, not the requested final 5k comparison.','',
           '| Retained/token | GIM 1k | Count 1k | Count 1,382 | GIM 2k | GIM 5k |','|---:|---:|---:|---:|---:|---:|']
    for k in curves['gim_1000']:
        lines.append('| '+str(k)+' | '+' | '.join(f"{curves[n][k]['kl']:.5f}" for n,_,_ in SPECS)+' |')
    differences={}
    lines+=['','## Step-matched 1k comparison','',
            'Count minus GIM KL, with paired approximate 95% t intervals across 64 contexts. Negative favors count. Intervals do not include training-seed uncertainty.','',
            '| Retained/token | Difference | Paired 95% interval |','|---:|---:|---:|']
    for k in [256,512,1024,2048,4096,8192]:
        a=curves['gim_1000'][k];b=curves['count_1000'][k]
        ds=[y-x for x,y in zip(a['sequence_kl'],b['sequence_kl'],strict=True)];assert len(ds)==64
        m=st.mean(ds);h=1.99834*st.stdev(ds)/math.sqrt(len(ds))
        differences[k]=dict(delta=m,interval=[m-h,m+h])
        lines.append(f'| {k} | {m:+.5f} | [{m-h:+.5f}, {m+h:+.5f}] |')
    thresholds={n:{str(t):min((k for k,r in c.items() if r['kl']<=t),default=None) for t in [.1,.3,1.]} for n,c in curves.items()}
    lines+=['','## Smallest tested counts crossing each KL target','', '| Checkpoint | KL ≤0.1 | KL ≤0.3 | KL ≤1.0 |','|---|---:|---:|---:|']
    for n,_,_ in SPECS:lines.append('| '+n+' | '+' | '.join(str(x) for x in thresholds[n].values())+' |')
    lines+=['','No interpolation or monotonicity assumption is used for these crossings.','']
    save_json(dict(records=records,step_matched_differences=differences,thresholds=thresholds),out/'comparison.json')
    (out/'report.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)
    receipts=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.suffix in ['.json','.md'] and 'wandb' not in p.parts and p.name not in ['complete.json','r2_receipts.json']:
            receipts.append(upload(p,'experiments/gim_count67_20260914/early_1382/'+str(p.relative_to(out))))
    receipts.append(upload(Path(__file__),'experiments/gim_count67_20260914/early_1382/gim_count67_early.py'))
    save_json(receipts,out/'r2_receipts.json')
    save_json(dict(complete=True,job_id=os.environ['SLURM_JOB_ID'],comparison_sha256=sha(out/'comparison.json'),verified_uploads=len(receipts)),out/'complete.json')
    upload(out/'complete.json','experiments/gim_count67_20260914/early_1382/complete.json')
    print('GIM_COUNT67_EARLY_COMPLETE',flush=True)


if __name__=='__main__':main()
