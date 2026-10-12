"""Summarize saved diagnostic JSON; all uncertainty units are independent contexts."""
import argparse
import json
import math
from pathlib import Path
import statistics as st

NAMES={'gim':'GIM','fisher':'Fisher K4','gaussian':'Gaussian','boundary001':'Gaussian + boundaries','vpd_5000':'VPD 5k','vpd_published':'Published VPD'}

def mean(xs): return st.mean(xs)
def paired_interval(a,b):
    diffs=[x-y for x,y in zip(a,b,strict=True)]
    m=mean(diffs); se=st.stdev(diffs)/math.sqrt(len(diffs))
    # Student t, two-sided 95%, df=15 for the predeclared 16-context pilot.
    t=2.131449545559323 if len(diffs)==16 else 1.96
    return m, m-t*se, m+t*se

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    data=json.loads((args.root/'diagnostic.json').read_text()); methods=data['methods']
    def get(n, budget, *keys):
        result=[]
        for b in methods[n]['batches']:
            item=b['budgets'][str(budget)]
            for k in keys:item=item[k]
            result.extend(item)
        return result
    lines=['# Why the Gaussian 67M pilot did not improve sparse reconstruction','',
           f"Explorer H200 diagnostic job {data['job_id']}. Frozen {data['parameters']:,}-parameter target. "
           f"{data['sequences']} validation contexts starting at {data['start_sequence']}, 512 positions each. "
           'All interventions are diagnostic only; no parameters were trained.','',
           'The 16 contexts are a small follow-up sample; paired intervals use contexts, not individual tokens. '
           'Exact FP32 tangent/path tests use only the first two contexts. '
           'The 64-probe ranking is still an estimator, not an optimal selector.','',
           '## Ranking precision','',
           '| Checkpoint | KL @512, K8 | KL @512, K64 | KL @512, native CI | K8/K64 overlap | KL @4096, K64 |','|---|---:|---:|---:|---:|---:|']
    summary={}
    for n in methods:
        if not methods[n]['batches']:continue
        vals={k:mean(get(n,512,'actual',k)) for k in methods[n]['batches'][0]['budgets']['512']['actual']}
        overlap=mean(get(n,512,'overlap_8_64'))
        lines.append(f"| {NAMES[n]} | {vals['fisher8']:.4f} | {vals['fisher64']:.4f} | {vals.get('native',float('nan')):.4f} | {overlap:.1%} | {mean(get(n,4096,'actual','fisher64')):.4f} |")
        summary[n]=dict(ranking512=vals,overlap512=overlap)
    lines+=['','Paired K64 − K8 mean KL differences at 512 (95% context-level t intervals):','']
    for n in summary:
        d,lo,hi=paired_interval(get(n,512,'actual','fisher64'),get(n,512,'actual','fisher8'))
        lines.append(f'- {NAMES[n]}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}].')
    lines+=['','## Concentration and layer interactions at 512','',
            'Ours uses Fisher K64 here; VPD uses native CI. “Restore L” keeps that layer intact and sparsifies the other three. '
            '“Only L” sparsifies that layer and keeps the other three intact. These are intervention contrasts, not additive causal shares.','',
            '| Checkpoint | Retained Fisher L1 mass | Retained Fisher squared mass | KL, all layers | Only L0 | Only L1 | Only L2 | Only L3 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in summary:
        rank=methods[n]['batches'][0]['budgets']['512']['main_rank']
        only=[mean(get(n,512,'only_layer',str(l))) for l in range(4)]
        restored=[mean(get(n,512,'restore_layer',str(l))) for l in range(4)]
        summary[n].update(only_layer512=only,restore_layer512=restored)
        lines.append('| '+NAMES[n]+' | '+ ' | '.join(f'{v:.4f}' for v in [mean(get(n,512,'retained_score_mass64')),mean(get(n,512,'retained_score_energy64')),mean(get(n,512,'actual',rank)),*only])+' |')
    lines+=['','| Checkpoint | Restore L0 | Restore L1 | Restore L2 | Restore L3 | Frozen clean-input error injection |','|---|---:|---:|---:|---:|---:|']
    for n in summary:
        lines.append('| '+NAMES[n]+' | '+' | '.join(f'{v:.4f}' for v in [*summary[n]['restore_layer512'],mean(get(n,512,'frozen_local_errors_actual'))])+' |')
    lines+=['','Frozen error injection adds each clean-input local replacement error at its matrix output; live factor replacements recompute errors on changed inputs. Both can have nonlinear downstream effects.','',
            '## Matrices that cause the most damage individually at 512','']
    for n in summary:
        keys=methods[n]['batches'][0]['budgets']['512']['only_matrix'].keys()
        worst=sorted(((mean(get(n,512,'only_matrix',k)),k) for k in keys),reverse=True)[:5]
        lines.append(f"- {NAMES[n]}: "+'; '.join(f'{k}: {v:.4f}' for v,k in worst)+'.')
        summary[n]['worst_matrices512']=worst
    lines+=['','## Exact full-sequence Fisher prediction','',
            'Quadratic KL is half the exact logit tangent Fisher norm, normalized by prediction positions. '
            'It includes cross-matrix and cross-source-position terms, including VPD’s omitted delta. '
            'The actual path replaces each target matrix output by `(1-alpha)*target + alpha*selected factors` on the live input.','',
            '| Checkpoint | Keep | Quadratic KL coefficient | KL at alpha .01 / .01² | KL at alpha .1 / .1² | KL at alpha .5 | KL at alpha 1 | Finite-difference relative error |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in summary:
        for k in [512,4096]:
            e=methods[n]['batches'][0]['budgets'][str(k)]['exact_path']
            vals=[mean(e['quadratic_kl']),mean(e['path']['0.01'])/.0001,mean(e['path']['0.1'])/.01,mean(e['path']['0.5']),mean(e['path']['1.0']),mean(e['finite_difference_relative_error'])]
            lines.append(f'| {NAMES[n]} | {k} | '+' | '.join(f'{v:.5g}' for v in vals)+' |')
    lines+=['','## Numerical and provenance checks','']
    for n in summary:
        b=methods[n]['batches'][0]
        lines.append(f"- {NAMES[n]}: all-kept FP32 KL {mean(b['all_kept']['fp32']):.4g}; BF16 KL {mean(b['all_kept']['bf16']):.4g}; maximum relative matrix reconstruction error {max(methods[n]['matrix_reconstruction_errors']):.4g}.")
        if 'native_positive_count' in b:
            positive=[v for x in methods[n]['batches'] for v in x['native_positive_count']]
            lines.append(f'  Native positive CI: {mean(positive):.1f} components/token; extra top-k slots may therefore retain zero-CI components.')
    lines+=['',f"Target SHA256: `{data['target_sha256']}`.",f"Data SHA256: `{data['data_sha256']}`.",f"Diagnostic source SHA256: `{data['source_sha256']}`.",
            '', 'Full per-context results, checkpoint hashes, local matrix errors, and retained counts are in `diagnostic.json`.','']
    (args.root/'report.md').write_text('\n'.join(lines))
    (args.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
