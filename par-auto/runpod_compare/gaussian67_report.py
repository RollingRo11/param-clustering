"""Summarize measured curves; do not interpolate crossings or assume monotonicity."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); args=ap.parse_args()
    records={}
    for name,relative in [('GIM 5k','controls/gim_5000'),('Fisher K4 5k','controls/fisher_5000'),
                          ('VPD 5k','controls/vpd_5000'),('Published VPD','controls/vpd_published'),
                          ('Gaussian Fisher 5k','gaussian/eval_fisher_k8'),('Gaussian + boundaries 5k','boundary001/eval_fisher_k8')]:
        p=args.root/relative/'klkeep.json'
        if p.exists():records[name]=json.loads(p.read_text())
    assert records, 'No evaluation results'
    signatures={(v['data_sha256'],v['positions'],v['start_sequence'],v['target_sha256']) for v in records.values()}
    assert len(signatures)==1, 'Mismatched targets or evaluation contexts'
    rows=['# Gaussian probe experiment on the 67M model','',
          'Explorer H200 runs; 5,000 of the existing 8,700 scheduled steps. All methods evaluated on validation contexts 256-383 (65,536 prediction positions).',
          'Our checkpoints share Fisher K=8 ranking and probe seeds; VPD uses its native CI. Counts are retained rank-one pieces, not groups.',
          '', '| Method | KL @256 | KL @512 | KL @1024 | KL @4096 | All-kept KL |', '|---|---:|---:|---:|---:|---:|']
    for name,v in records.items():
        by_k={int(r['kept_components']):r for r in v['curve']}
        values=[by_k[k]['kl'] for k in [256,512,1024,4096,v['total_components']]]
        rows.append('| '+name+' | '+' | '.join(f'{x:.6g}' for x in values)+' |')
    rows+=['','Smallest **tested** retained counts meeting each mean-KL target; no interpolation or monotonicity assumption.','',
           '| Method | KL <=0.1 | KL <=0.3 | KL <=1.0 |','|---|---:|---:|---:|']
    thresholds={}
    for name,v in records.items():
        thresholds[name]={str(t):min((int(r['kept_components']) for r in v['curve'] if r['kl']<=t),default=None) for t in [.1,.3,1.]}
        rows.append('| '+name+' | '+' | '.join(str(x) if x is not None else 'not reached' for x in thresholds[name].values())+' |')
    prep=args.root/'prepare.json'
    if prep.exists():
        timing=json.loads(prep.read_text());rows+=['','Complete-update benchmark (mean of two 25-update measured windows after warmup):','']
        rows += [f'- {n}: {v:.4f} s/update ({timing["relative_to_gim"][n]:.3f}x GIM).' for n,v in timing['mean_seconds'].items()]
    rows+=['','Single-seed pilot. The Gaussian moment identity is exact for the unsmoothed objective; this controlled run retains the existing epsilon-smoothed loss.',
           'The boundary coefficient is 0.01; boundary scales are fixed RMS vector norms estimated from four training contexts.',
           'No component masks or model perturbations were used during training. Masks are used only for evaluation.','']
    (args.root/'report.md').write_text('\n'.join(rows))
    (args.root/'comparison.json').write_text(json.dumps(dict(results=records,thresholds=thresholds),indent=2)+'\n')
    print('\n'.join(rows))


if __name__=='__main__':main()
