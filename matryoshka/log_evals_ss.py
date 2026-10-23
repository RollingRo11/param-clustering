"""Log a SimpleStories run's KLKeep and adversarial-probe results to W&B project matryoshka-evals (one run per training
run, same id and name; the training dashboard stays training-only), under the names the
earlier SS runs used (log_evals.py of the 20 Oct runs): summary eval/klkeep/kl_<fraction> and eval/adv/<gate>/step_<n>,
line plots eval/klkeep_curve and eval/adv_curve_<gate>, all ranked by Fisher K=8; KLKeep ranked by the cached
text-averaged Fisher and by write size under eval/klkeep_static/... and eval/klkeep_write/...; a run trained with one of
those also gets its probe under eval/adv_<sensor>/... Writes evals_logged.json.
Usage: log_evals_ss.py RUN_DIR"""
import json
import os
import sys
from pathlib import Path

import wandb

run_dir = Path(sys.argv[1])
# (key suffix, KLKeep, adversarial probe or None): Fisher K=8 under the old names; the other rankings beside it.
plan = json.loads((run_dir/'plan.json').read_text())
sources = [('', run_dir/'eval_fisher_k8/klkeep.json', run_dir/'adversarial_global_128.json')]
for s in ('static', 'write'):
    probe = run_dir/f'adversarial_global_128_{s}.json'
    sources.append((f'_{s}', run_dir/f'eval_{s}/klkeep.json', probe if plan['sensor'] == s else None))
if plan.get('score_input', {}).get('kind') == 'masked':
    sources.append((f'_{plan["sensor"]}_masked', run_dir/f'eval_{plan["sensor"]}_masked/klkeep.json', None))
missing = [str(p) for _, *paths in sources for p in paths if p is not None and not p.exists()]
if missing: sys.exit(f'evals not ready: {missing}')
training = json.loads((run_dir/'wandb.json').read_text())
# One run per training run in project matryoshka-evals, kept off the training dashboard; same id and name.
name = wandb.Api().run(f'rohan-kathuria-neu/matryoshka/{training["id"]}').name
run = wandb.init(id=training['id'], resume='allow', project='matryoshka-evals', entity='rohan-kathuria-neu', name=name,
                 job_type='evaluation', config=dict(training_run=training['url'], sensor=plan['sensor']),
                 dir=os.environ.get('MATRYOSHKA_WANDB_DIR', '/root/wandb_eval'))
summary, charts = {}, {}
for suffix, klkeep, adversarial in sources:
    k = json.loads(klkeep.read_text())
    summary.update({f'eval/klkeep{suffix}/kl_{c["keep_fraction"]:g}': c['kl'] for c in k['curve']})
    table = wandb.Table(columns=['kept_components', 'kl'], data=[[c['kept_components'], c['kl']] for c in k['curve'] if c['kept_components'] > 0])
    charts[f'eval/klkeep{suffix}_curve'] = wandb.plot.line(table, 'kept_components', 'kl', title=f'KLKeep, {k["sensor"]}')
    if adversarial is None: continue
    a = json.loads(adversarial.read_text())
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
