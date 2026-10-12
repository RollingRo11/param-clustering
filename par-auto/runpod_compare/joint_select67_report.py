"""Summarize frozen support selection; completion requires verified archival."""
import json
import math
from pathlib import Path
import statistics as st
from common import save_json, upload, sha


def finish(out):
    plan = json.loads((out/'plan.json').read_text())
    lines = ['# Joint selection on frozen 67M decompositions', '',
             f"Explorer; {plan['sequences']} validation contexts beginning at {plan['first']}. "
             f"Fit Fisher K{plan['fit_k']}, independent check Fisher K{plan['check_k']}; "
             'FP32 signed score arithmetic and BF16 model forwards. No factor training.', '',
             'Binary support swaps preserve the count at every token and leave all weights fixed. '
             'Two starts; choose the final support using fit quadratic only. The bounded local search '
             'is not an optimal selector. Intervals quantify context variation, not training-seed uncertainty.', '',
             '| Method | Count/token | Magnitude KL | Selected KL | Difference [95% paired interval] | Fit Q before → after | Independent Q before → after |',
             '|---|---:|---:|---:|---|---|---|']
    result = dict(plan=plan, methods={})
    for name in ['gim', 'count', 'vpd']:
        rows = [json.loads((out/f'{i}_{name}.json').read_text())
                for i in range(plan['first'], plan['first']+plan['sequences'])]
        method = result['methods'][name] = dict(records=rows, budgets={})
        for budget in plan['budgets']:
            bs = [r['budgets'][str(budget)] for r in rows]
            avg = lambda variant, key: st.mean(b[variant][key] for b in bs)
            ds = [b['selected']['kl']-b['magnitude']['kl'] for b in bs]
            m = st.mean(ds)
            # Protocol is fixed to n=16. For other n, do not report a fabricated t interval.
            h = 2.13145*st.stdev(ds)/math.sqrt(len(ds)) if len(ds)==16 else None
            cell = f'{m:+.5f} [{m-h:+.5f}, {m+h:+.5f}]' if h is not None else f'{m:+.5f} (pilot; no interval)'
            entry = dict(magnitude_kl=avg('magnitude', 'kl'), selected_kl=avg('selected', 'kl'),
                         delta=m, interval=None if h is None else [m-h, m+h],
                         fit_before=avg('magnitude', 'fit_quadratic'), fit_after=avg('selected', 'fit_quadratic'),
                         independent_before=avg('magnitude', 'independent_quadratic'),
                         independent_after=avg('selected', 'independent_quadratic'),
                         magnitude_restore_last_kl=avg('magnitude', 'restore_last_kl'),
                         selected_restore_last_kl=avg('selected', 'restore_last_kl'),
                         overlap=st.mean(b['overlap_with_magnitude'] for b in bs))
            if name == 'vpd':
                entry['native_kl'] = avg('native', 'kl')
                entry['native_independent_quadratic'] = avg('native', 'independent_quadratic')
            method['budgets'][str(budget)] = entry
            lines.append(f"| {name} | {budget} | {entry['magnitude_kl']:.5f} | {entry['selected_kl']:.5f} | {cell} | "
                         f"{entry['fit_before']:.4f} → {entry['fit_after']:.4f} | {entry['independent_before']:.4f} → {entry['independent_after']:.4f} |")
    lines += ['', '## Native VPD control', '', '| Count/token | Native CI KL |', '|---:|---:|']
    for budget, entry in result['methods']['vpd']['budgets'].items():
        lines.append(f"| {budget} | {entry['native_kl']:.5f} |")
    lines += ['', '## Last-layer restoration', '',
              '| Method | Count/token | Magnitude KL change on restoration | Selected KL change on restoration |',
              '|---|---:|---:|---:|']
    for name, method in result['methods'].items():
        for budget, entry in method['budgets'].items():
            lines.append(f"| {name} | {budget} | {entry['magnitude_restore_last_kl']-entry['magnitude_kl']:+.5f} | "
                         f"{entry['selected_restore_last_kl']-entry['selected_kl']:+.5f} |")
    lines += ['', 'Clean-input errors per matrix, all-kept controls, both start results, '
              'search monotonicity and compressed exact supports are saved alongside this report. '
              'Fit-sketch improvement without independent/actual improvement is not a successful sparsification. '
              'Failure of this bounded search does not establish a factor-quality floor.', '']
    save_json(result, out/'comparison.json')
    (out/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)
    receipts = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.suffix in ['.json', '.md', '.npz'] and p.name not in ['complete.json', 'r2_receipts.json']:
            receipts.append(upload(p, 'experiments/joint_select67_20260915/'+p.name))
    for p in Path(__file__).parent.iterdir():
        if p.is_file():
            receipts.append(upload(p, 'experiments/joint_select67_20260915/source/'+p.name))
    save_json(receipts, out/'r2_receipts.json')
    save_json(dict(complete=True, contexts=plan['sequences'], verified_uploads=len(receipts),
                   comparison_sha256=sha(out/'comparison.json')), out/'complete.json')
    upload(out/'complete.json', 'experiments/joint_select67_20260915/complete.json')
    print('JOINT_SELECT67_COMPLETE', flush=True)
