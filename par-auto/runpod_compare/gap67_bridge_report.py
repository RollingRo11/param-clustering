import json
from pathlib import Path
import statistics as st
from common import save_json, sha, upload


def finish(out, plan, records):
    result = dict(plan=plan, records=records, native_count=st.mean(r['native_count_mean'] for r in records), methods={})
    lines = ['# Can the VPD solution fit GIM’s exact-sum factor family?', '',
             f"No training. Convert {plan['source_components']} published VPD pieces into exactly {plan['converted_components']} pieces, "
             'matching GIM’s capacity in every matrix. Fill extra slots with residual SVD factors, then apply FP64 exact-sum projection. '
             'Keep original component indices and map the existing CI support; new slots are absent from that support.', '',
             f"Eight contexts 424–431, FP32 actual KL. Native positive CI selects {result['native_count']:.2f} pieces/token on average; "
             'its per-token variable budgets are copied exactly to the corresponding Fisher controls.', '',
             '| Factors | Support | KL | Mean pieces/token |', '|---|---|---:|---:|']
    for method in records[0]['methods']:
        ms = [r['methods'][method] for r in records]
        summary = result['methods'][method] = dict(variants={}, objective={})
        for variant in ms[0]['variants']:
            vs = [m['variants'][variant] for m in ms]
            entry = {k:st.mean(v[k] for v in vs) for k in ['kl','count_mean','clean_error_injection_kl','restore_last_kl']}
            summary['variants'][variant] = entry
            lines.append(f"| {method} | {variant} | {entry['kl']:.5f} | {entry['count_mean']:.2f} |")
        summary['objective'] = {k:st.mean(m['objective'][k] for m in ms) for k in ms[0]['objective']}
        summary['all_kept_kl'] = st.mean(m['all_kept_kl'] for m in ms)
    lines += ['', '## Common normalized training objectives', '',
              'Same GIM initial reference normalizers for every candidate. Actual BF16 GIM scoring recipe; '
              'the converted VPD and GIM candidates have identical per-matrix capacity and exact sums. '
              'These are measured objectives, not training outcomes. Original VPD still has an omitted Delta.', '',
              '| Factors | Lp p=.5 | Lp p=.641 | Lp p=1 | Soft count | Normalized count | Lp+count p=.641 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method, m in result['methods'].items():
        v=m['objective']
        lines.append('| '+method+' | '+' | '.join(f'{v[k]:.5f}' for k in ['0.5','0.641','1.0','soft_count','normalized_count','lp_plus_count_p641'])+' |')
    lines += ['', 'The conversion is an existence/constraint diagnostic. It uses VPD-trained factors and its selector; '
              'it does not constitute a new method trained without masks. If the sparse endpoint survives but our loss is higher, '
              'our loss ranks this known sparse solution worse than GIM. That does not prove all gradient-based objectives fail.', '']
    save_json(result, out/'comparison.json'); (out/'report.md').write_text('\n'.join(lines)); print('\n'.join(lines), flush=True)
    receipts=[]
    for p in out.iterdir():
        if p.is_file() and p.name not in ['complete.json','r2_receipts.json']:
            receipts.append(upload(p, 'experiments/gap67_bridge_20260915/'+p.name))
    for p in Path(__file__).parent.iterdir():
        if p.is_file():
            receipts.append(upload(p, 'experiments/gap67_bridge_20260915/source/'+p.name))
    save_json(receipts,out/'r2_receipts.json')
    save_json(dict(complete=True,verified_uploads=len(receipts),comparison_sha256=sha(out/'comparison.json')),out/'complete.json')
    upload(out/'complete.json','experiments/gap67_bridge_20260915/complete.json')
    print('GAP67_BRIDGE_COMPLETE',flush=True)
