"""Small-result analysis only: locked discovery/confirmation mixture screen.

No torch, model/data loads, or new factor training. Can run locally after fetching
objective67's JSONs. Positive mixtures use at most two physical error metrics.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics as st


def value(record, state, selector, metric):
    return record['states'][state]['variants'][selector]['conditional'][metric]


def pairs(record):
    witness = ('vpd_exact', 'inherited_native')
    candidates = [witness, ('gim5k', 'fisher'), ('vpd_exact', 'fisher'),
                  ('vpd_exact', 'random_native_allocation'),
                  ('descent_20', 'inherited_native'), ('descent_100', 'inherited_native'),
                  ('descent_200', 'inherited_native')]
    # Include the witness comparisons and adjacent points on the controlled
    # trajectory. Orient each pair using its measured KL, not the state label.
    links = [(witness, other) for other in candidates[1:]]
    links += [(candidates[-3], candidates[-2]), (candidates[-2], candidates[-1])]
    result = []
    for a, b in links:
        ka = record['states'][a[0]]['variants'][a[1]]['kl']
        kb = record['states'][b[0]]['variants'][b[1]]['kl']
        if abs(ka-kb) < .02:
            continue
        result.append((a, b) if ka < kb else (b, a))
    return result


def assess(rows, weights, scales):
    good = total = 0
    soft = []
    ratios = []
    comparisons = []
    for record in rows:
        def loss(state, selector):
            return sum(w*value(record, state, selector, m)/scales[m] for m, w in weights.items())
        native = loss('vpd_exact', 'inherited_native')
        gim = loss('gim5k', 'fisher')
        ratios.append(native/max(gim, 1e-30))
        for a, b in pairs(record):
            la, lb = loss(*a), loss(*b)
            # Bounded, scale-stable signed margin, positive means correct order.
            margin = (lb-la)/(abs(la)+abs(lb)+1e-12)
            good += margin > 0
            total += 1
            soft.append(margin)
            comparisons.append(dict(sequence=record['sequence'], better_kl_state=list(a),
                                    worse_kl_state=list(b), relative_loss_margin=margin))
    return dict(pairwise_accuracy=good/total if total else None, pairs=total,
                mean_relative_margin=st.mean(soft), mean_vpd_gim_ratio=st.mean(ratios),
                vpd_preferred_fraction=st.mean(float(x<1) for x in ratios), comparisons=comparisons)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('output', type=Path); args = ap.parse_args()
    data = json.loads((args.output/'comparison.json').read_text())
    records = data['records']
    discovery = [r for r in records if r['split']=='discovery']
    confirmation = [r for r in records if r['split']=='confirmation']
    metrics = list(data['conditional'])
    # Fixed scales learned on discovery only; no per-checkpoint normalization.
    scales = {m:max(1e-12,st.mean(value(r,'gim5k','fisher',m) for r in discovery)) for m in metrics}
    candidates = [{m:1.} for m in metrics]
    for a,b in itertools.combinations(metrics,2):
        for weight in [.1,.25,.5,.75,.9]:
            candidates.append({a:weight,b:1-weight})
    ranked = [(assess(discovery,w,scales),w) for w in candidates]
    ranked.sort(key=lambda x:(-x[0]['pairwise_accuracy'],-x[0]['mean_relative_margin'],len(x[1]),str(x[1])))
    eligible = [(r,w) for r,w in ranked if r['mean_vpd_gim_ratio']<1 and r['vpd_preferred_fraction']>=.75]
    selected_discovery, selected = (eligible or ranked)[0]
    confirmed = assess(confirmation,selected,scales)
    dominance = {split:dict(
        all_conditional_losses_reject_vpd=all(st.mean(value(r,'vpd_exact','inherited_native',m) for r in rows)>
                                             st.mean(value(r,'gim5k','fisher',m) for r in rows) for m in metrics),
        all_maskfree_losses_reject_vpd=all(v[split]['ratio']>1 for v in data['maskfree'].values()))
        for split,rows in [('discovery',discovery),('confirmation',confirmation)]}
    timing = {}
    for key,p in data['probes'].items():
        projection = st.median(x['projection_seconds'] for x in p['steps'])
        timing[key] = dict(cache=p['cache_seconds'],factor_forward_backward=p['factor_forward_backward_seconds'],
                           projection=projection,total_proxy=p['cache_seconds']+p['factor_forward_backward_seconds']+projection)
    baseline_time = timing.get('gim_0.641',{}).get('total_proxy')
    if baseline_time:
        for row in timing.values():row['relative_to_gim']=row['total_proxy']/baseline_time
    result = dict(candidate_count=len(candidates),eligible_count=len(eligible),selected_accepts_vpd_on_discovery=bool(eligible),
                  objective_type='subset-conditioned diagnostic mixture',
                  selection='discovery only; require mean VPD/GIM ratio <1 and preference in >=75% of contexts, then maximize pairwise accuracy and margin; if none qualify, label best diagnostic as unaccepted',
                  scales=scales,selected_weights=selected,discovery=selected_discovery,confirmation=confirmed,
                  dominance=dominance,timing_proxy=timing,
                  caveat='Four confirmation contexts; not a selector, production-cost benchmark, or trained sparse method.')
    (args.output/'mixtures.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Objective-screen analysis','',
           f"Tested {len(candidates)} single/two-term nonnegative combinations of the 12 subset-conditioned diagnostics. "
           'Chosen using discovery only, with fixed discovery-derived scales. KL orders pairs for this small diagnostic-fitting exercise; '
           'it never provides a gradient to factor training.','',
           f"{len(eligible)} candidates pass the discovery acceptance gate (mean VPD/GIM ratio below one and VPD preferred in at least 75% of contexts). "
           +('The selected mixture passes that gate.' if eligible else 'None passes; the combination below is only the best diagnostic ordering, not an accepted solution.'),'',
           'Selected weights: `'+json.dumps(selected,sort_keys=True)+'`. Divide each constituent by its recorded fixed scale before weighting.','',
           '| Split | Pairwise ordering accuracy | Mean VPD/GIM loss ratio | Context fraction favoring VPD |',
           '|---|---:|---:|---:|',
           f"| Discovery | {selected_discovery['pairwise_accuracy']:.3f} | {selected_discovery['mean_vpd_gim_ratio']:.3f} | {selected_discovery['vpd_preferred_fraction']:.2f} |",
           f"| Confirmation | {confirmed['pairwise_accuracy']:.3f} | {confirmed['mean_vpd_gim_ratio']:.3f} | {confirmed['vpd_preferred_fraction']:.2f} |",'',
           'Pairs are within contexts and include native-versus-Fisher/random/GIM and the controlled damage trajectory. '
           'Pairs separated by less than 0.02 actual KL are omitted. Correlated pairs are not independent samples; '
           'the validation unit is the context. This is a hypothesis screen with four confirmation contexts, not a significance claim.','',
           '## Timing proxy','',
           'Single-context intact capture + factor forward/backward + one exact-sum projection, at the common diagnostic precision. '
           'Optimizer updates and production-batch throughput are not measured. Projection times use the median of three proposals. '
           'Treat small differences as timing noise. Forward-only trace capture also includes attention-reproduction validation, so that cache cost is conservative. '
           'Conditional-suite timing cannot be assigned to individual losses.','',
           '| Objective | Proxy seconds | Relative to GIM p=.641 | Selected-step KL change |','|---|---:|---:|---:|']
    for k,row in sorted(timing.items(),key=lambda x:x[1]['total_proxy']):
        p=data['probes'][k]['selected'];delta='no descent' if p is None else f"{p['heldout_kl_mean']-data['baseline_heldout_kl']:+.5f}"
        lines.append(f"| {k} | {row['total_proxy']:.4f} | {row.get('relative_to_gim',float('nan')):.3f} | {delta} |")
    lines+=['','## Nonnegative-mixture obstruction','',
            'If every constituent loss has a larger mean on VPD than on GIM, every nonzero nonnegative linear combination does too. '
            'This follows by summing positive differences. Negative weights can reverse it but would reward increased error in at least one term; '
            'they are excluded from this screen.','', '`'+json.dumps(dominance,sort_keys=True)+'`','']
    (args.output/'analysis.md').write_text('\n'.join(lines))
    print(json.dumps(dict(candidate_count=len(candidates),selected_weights=selected,
                         confirmation={k:v for k,v in confirmed.items() if k!='comparisons'},dominance=dominance),indent=2))


if __name__=='__main__':main()
