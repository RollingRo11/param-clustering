"""Attach a SimpleStories run's KLKeep and adversarial-probe results to its W&B training run, under the names the
earlier SS runs used (log_evals.py of the 20 Oct runs): summary eval/klkeep/kl_<fraction> and eval/adv/<gate>/step_<n>,
line plots eval/klkeep_curve and eval/adv_curve_<gate>, all ranked by Fisher K=8. A run ranked by another sensor also
gets the same keys under eval/klkeep_<sensor>/..., eval/adv_<sensor>/... (its own ranking). Writes evals_logged.json.
Usage: log_evals_ss.py RUN_DIR"""
import json
import os
import sys
from pathlib import Path

import wandb

run_dir = Path(sys.argv[1])
sources = [('', run_dir/'eval_fisher_k8/klkeep.json', run_dir/'adversarial_global_128.json')]
plan = json.loads((run_dir/'plan.json').read_text())
if plan['sensor'] != 'fisher':
    s = plan['sensor']
    sources.append((f'_{s}', run_dir/f'eval_{s}/klkeep.json', run_dir/f'adversarial_global_128_{s}.json'))
missing = [str(p) for _, *paths in sources for p in paths if not p.exists()]
if missing: sys.exit(f'evals not ready: {missing}')
run = wandb.init(id=json.loads((run_dir/'wandb.json').read_text())['id'], resume='must', project='matryoshka',
                 entity='rohan-kathuria-neu', dir=os.environ.get('MATRYOSHKA_WANDB_DIR', '/root/wandb_eval'))
summary, charts = {}, {}
for suffix, klkeep, adversarial in sources:
    k, a = json.loads(klkeep.read_text()), json.loads(adversarial.read_text())
    summary.update({f'eval/klkeep{suffix}/kl_{c["keep_fraction"]:g}': c['kl'] for c in k['curve']})
    table = wandb.Table(columns=['kept_components', 'kl'], data=[[c['kept_components'], c['kl']] for c in k['curve'] if c['kept_components'] > 0])
    charts[f'eval/klkeep{suffix}_curve'] = wandb.plot.line(table, 'kept_components', 'kl', title=f'KLKeep, {k["sensor"]}')
    for gate, curve in a['kl'].items():
        gate = gate.replace('top', 'keep_')
        for step, value in curve.items(): summary[f'eval/adv{suffix}/{gate}/step_{step}'] = value
        rows = [[int(s), v] for s, v in curve.items() if s.isdigit()]
        charts[f'eval/adv{suffix}_curve_{gate}'] = wandb.plot.line(wandb.Table(columns=['attack_steps', 'kl'], data=rows), 'attack_steps', 'kl',
                                                                   title=f'Adversarial KL, {gate}, {a["ranking"]} ranking')
run.log(charts)
run.summary.update(summary)
run.finish()
(run_dir/'evals_logged.json').write_text(json.dumps(dict(run=run.url, keys=sorted(summary)), indent=2) + '\n')
print('LOGGED', run_dir.name, run.url)
