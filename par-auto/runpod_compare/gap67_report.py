"""Report the fixed-factor endpoint and finite-response controls."""
import json
import math
from pathlib import Path
import statistics as st
from common import save_json, sha, upload


def summarize(out):
    plan = json.loads((out/'plan.json').read_text())
    result = dict(plan=plan, methods={})
    lines = ['# Locating the GIM–VPD gap on the frozen 67M model', '',
             f"Explorer, FP32 evaluation, {plan['sequences']} validation contexts {plan['first']}–{plan['first']+plan['sequences']-1}. "
             'All counts are fixed rank-one pieces per token across 24 matrices.', '',
             'Endpoint search uses gradients of actual sparse-model KL and accepts only actual improvements. '
             'Weights remain fixed. Two starts, at most 32 rounds; retain the best binary support including the initial magnitude support. '
             'This per-context search consults target outputs and is a diagnostic, not a trained selector or a proven optimum. '
             'VPD 5k matches update count, not total compute or warmup. Published VPD is trained much longer.', '',
             '| Factors | Count/token | Magnitude KL | Best endpoint KL | Difference [paired 95% interval] | Native VPD KL |',
             '|---|---:|---:|---:|---|---:|']
    for name in plan['checkpoints']:
        summary = result['methods'][name] = dict(budgets={}, records=[])
        for k in plan['budgets']:
            rs = [json.loads((out/f'{i}_{name}_{k}.json').read_text())
                  for i in range(plan['first'], plan['first']+plan['sequences'])]
            summary['records'].extend(rs)
            before = [r['variants']['magnitude']['kl'] for r in rs]
            after = [r['variants'][r['best_endpoint']]['kl'] for r in rs]
            ds = [b-a for a,b in zip(before, after, strict=True)]
            h = 2.36462*st.stdev(ds)/math.sqrt(8) if len(ds)==8 else None
            entry = dict(magnitude_kl=st.mean(before), endpoint_kl=st.mean(after), delta=st.mean(ds),
                         interval=None if h is None else [st.mean(ds)-h, st.mean(ds)+h], variants={})
            for label in rs[0]['variants']:
                entry['variants'][label] = {key: st.mean(r['variants'][label][key] for r in rs)
                                           for key in ['kl', 'clean_error_injection_kl', 'restore_last_kl']}
                paths = [r['variants'][label] for r in rs if 'exact_quadratic' in r['variants'][label]]
                if paths:
                    entry['variants'][label]['exact_quadratic'] = st.mean(p['exact_quadratic'] for p in paths)
                    entry['variants'][label]['exact_path_contexts'] = len(paths)
            entry['best_injection_kl'] = st.mean(r['variants'][r['best_endpoint']]['clean_error_injection_kl'] for r in rs)
            entry['best_restore_kl'] = st.mean(r['variants'][r['best_endpoint']]['restore_last_kl'] for r in rs)
            entry['all_kept_kl'] = st.mean(r['all_kept_kl'] for r in rs)
            summary['budgets'][str(k)] = entry
            native = f"{entry['variants']['native']['kl']:.5f}" if 'native' in entry['variants'] else '—'
            ci = f"[{entry['interval'][0]:+.5f}, {entry['interval'][1]:+.5f}]" if h is not None else '(pilot)'
            lines.append(f"| {name} | {k} | {entry['magnitude_kl']:.5f} | {entry['endpoint_kl']:.5f} | {entry['delta']:+.5f} {ci} | {native} |")
    lines += ['', '## Finite response to upstream errors', '',
              'Clean-error injection adds each selected matrix’s clean-input error to the original matrix output '
              'during an otherwise live forward. It removes the ΔW·δh term while retaining the original matrix response W·δh. '
              'The contrast is not an additive decomposition of KL.', '',
              '| Factors | Count/token | Support | Actual KL | Clean-error injection KL | Restore-last KL |',
              '|---|---:|---|---:|---:|---:|']
    for name, method in result['methods'].items():
        for k, entry in method['budgets'].items():
            for label, v in entry['variants'].items():
                lines.append(f"| {name} | {k} | {label} | {v['kl']:.5f} | {v['clean_error_injection_kl']:.5f} | {v['restore_last_kl']:.5f} |")
    lines += ['', '## Intact-model attribution objective census', '',
              'Raw sums over all primitive factors, averaged per token, with common sensors. '
              'VPD’s Delta is excluded here, so these are not objective comparisons between equally constrained feasible decompositions. '
              'The activity count is s/(s+0.001); Lp uses (s+1e-6)^p−1e-6^p.', '',
              '| Factors | Fisher Lp, p=.641 | GIM Lp, p=.641 | GIM soft count |', '|---|---:|---:|---:|']
    for name, method in result['methods'].items():
        rs = method['records']
        lines.append(f"| {name} | {st.mean(r['census']['fisher']['0.641'] for r in rs):.3f} | "
                     f"{st.mean(r['census']['gim']['0.641'] for r in rs):.3f} | "
                     f"{st.mean(r['census']['gim']['soft_count_tau_001'] for r in rs):.1f} |")
    lines += ['', 'Exact FP32 full-sequence tangents and interpolation paths are recorded for the first two contexts only. '
              'All eight contexts have finite-response, restoration, exact-support and search-history records. '
              'A failure to recover known-good native VPD supports limits conclusions about the search’s ability to diagnose a factor floor.', '']
    return result, '\n'.join(lines)


def finish(out):
    result, report = summarize(out)
    save_json(result, out/'comparison.json')
    (out/'report.md').write_text(report)
    print(report, flush=True)
    receipts = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.suffix in ['.json', '.md', '.npz'] and p.name not in ['complete.json', 'r2_receipts.json']:
            receipts.append(upload(p, 'experiments/gap67_20260915/'+p.name))
    for p in Path(__file__).parent.iterdir():
        if p.is_file():
            receipts.append(upload(p, 'experiments/gap67_20260915/source/'+p.name))
    save_json(receipts, out/'r2_receipts.json')
    save_json(dict(complete=True, verified_uploads=len(receipts), comparison_sha256=sha(out/'comparison.json')), out/'complete.json')
    upload(out/'complete.json', 'experiments/gap67_20260915/complete.json')
    print('GAP67_COMPLETE', flush=True)
