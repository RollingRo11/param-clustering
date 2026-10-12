"""Matched component-budget comparison; retain paired per-context uncertainty."""
import json
import math
from pathlib import Path
import statistics as st
from common import ROOT, save_json, sha, upload, HERE

OUT=ROOT/'output/gim_count67_20260914'


def main():
    records={n:json.loads((OUT/n/'klkeep.json').read_text()) for n in ['count_fisher64','gim_fisher64','count_gim','gim_gim']}
    assert len({(r['target_sha256'],r['data_sha256'],r['training_data_sha256'],r['positions'],r['start_sequence'],r['training_steps']) for r in records.values()})==1
    assert all(r['training_steps']==5000 and r['total_components']==43008 for r in records.values())
    prepared=json.loads((OUT/'prepare.json').read_text())
    lines=['# GIM plus count versus normal GIM on the 67M model','',
           'Fresh seed-0 training for 5,000 steps on Explorer. Same 163,840,000 training positions and 8,700-step schedule horizon as the normal GIM checkpoint. '
           'GIM attribution, existing Lp loss, and exact-sum projection are retained. Added count: s/(s+0.001), coefficient 1 after fixed global initial normalization. '
           'No masks or Gaussian probes were used in training.','',
           'Evaluation: 128 validation contexts (256–383), with exactly matched retained rank-one component counts. '
           'Primary ranking is common Fisher K64; native GIM ranking is a secondary comparison. '
           'Intervals below are approximate paired 95% intervals across contexts (1.98 standard errors); they do not account for training-seed variance.','']
    comparisons={}
    for sensor in ['fisher64','gim']:
        control=records['gim_'+sensor];trial=records['count_'+sensor]
        lines += [f'## {sensor} ranking','', '| Retained/token | Normal GIM KL | GIM + count KL | Difference | Paired 95% interval |', '|---:|---:|---:|---:|---:|']
        rows=[]
        for a,b in zip(control['curve'],trial['curve'],strict=True):
            assert a['kept_components']==b['kept_components']
            diff=[y-x for x,y in zip(a['sequence_kl'],b['sequence_kl'],strict=True)]
            assert len(diff)==128
            delta=st.mean(diff);half=1.98*st.stdev(diff)/math.sqrt(len(diff))
            row=dict(kept=int(a['kept_components']),control_kl=a['kl'],trial_kl=b['kl'],delta=delta,interval=[delta-half,delta+half])
            rows.append(row)
            lines.append(f"| {row['kept']} | {a['kl']:.5f} | {b['kl']:.5f} | {delta:+.5f} | [{delta-half:+.5f}, {delta+half:+.5f}] |")
        thresholds={name:{str(t):min((int(r['kept_components']) for r in data['curve'] if r['kl']<=t),default=None) for t in [.1,.3,1.]} for name,data in [('gim',control),('count',trial)]}
        comparisons[sensor]=dict(rows=rows,smallest_tested_count_at_kl=thresholds)
        lines+=['','Smallest tested counts crossing KL thresholds (no interpolation): `'+json.dumps(thresholds)+'`.','']
    lines+=['## Whole-update runtime','']
    lines += [f"- {name}: {value:.4f} seconds/update ({prepared['relative_to_gim'][name]:.3f}× normal GIM)." for name,value in prepared['mean_seconds'].items()]
    lines+=['','This is one fixed coefficient/threshold setting and one training seed. A lower soft count alone is not a win; actual KLKeep and retained counts decide the comparison.','']
    save_json(dict(comparisons=comparisons,profiles=prepared,records=records),OUT/'comparison.json')
    (OUT/'report.md').write_text('\n'.join(lines));print('\n'.join(lines))
    result=json.loads((OUT/'result.json').read_text());assert result['complete'] and result['completed']==5000
    receipts=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or 'wandb' in p.parts or p.name in ['complete.json','r2_receipts.json']:continue
        if p.name=='checkpoint.pt' and p.parent.name!='005000':continue
        if p.suffix in ['.json','.md','.pth','.pt']:
            receipts.append(upload(p,'experiments/gim_count67_20260914/'+str(p.relative_to(OUT))))
    import tarfile
    with tarfile.open(OUT/'source.tar.gz','w:gz') as archive:
        for p in sorted(HERE.parent.rglob('*')):
            if p.is_file() and p.suffix in ['.py','.yaml','.json','.sha256'] and '__pycache__' not in p.parts:archive.add(p,arcname=str(p.relative_to(HERE.parent)))
    receipts.append(upload(OUT/'source.tar.gz','experiments/gim_count67_20260914/source.tar.gz'))
    save_json(receipts,OUT/'r2_receipts.json')
    upload(OUT/'r2_receipts.json','experiments/gim_count67_20260914/r2_receipts.json')
    save_json(dict(complete=True,completed=5000,comparison_sha256=sha(OUT/'comparison.json'),verified_uploads=len(receipts)),OUT/'complete.json')
    upload(OUT/'complete.json','experiments/gim_count67_20260914/complete.json')
    print('GIM_COUNT67_COMPLETE_AND_ARCHIVED',flush=True)


if __name__=='__main__':main()
