"""Add the 20 Oct SimpleStories runs' curve names (matryoshka67.SS_ALIASES) to an SS training run on W&B that was
logged before matryoshka67.py logged them itself: resumes the run and logs, per update, train/<old name> = the
value already logged as train/<new name>. particles_r_saturated has no counterpart to copy. Run it once the
training job has finished (W&B allows one writer). Usage: backfill_ss_aliases.py RUN_DIR"""
import json
import os
import sys
from pathlib import Path

import wandb

os.environ.setdefault('MATRYOSHKA_TARGET', 'simplestories')
from matryoshka67 import SS_ALIASES  # noqa: E402

run_dir = Path(sys.argv[1])
run_id = json.loads((run_dir/'wandb.json').read_text())['id']
history = wandb.Api(timeout=120).run(f'rohan-kathuria-neu/matryoshka/{run_id}')
keys = ['update'] + [f'train/{new}' for new in set(SS_ALIASES.values())]
rows = {}
for row in history.scan_history(page_size=2000):
    if row.get('update') is None: continue
    for old, new in SS_ALIASES.items():
        if row.get(f'train/{new}') is not None and f'train/{old}' not in row:
            rows.setdefault(int(row['update']), {})[f'train/{old}'] = row[f'train/{new}']
run = wandb.init(id=run_id, resume='must', project='matryoshka', entity='rohan-kathuria-neu',
                 dir=os.environ.get('MATRYOSHKA_WANDB_DIR', '/root/wandb_eval'))
run.define_metric('update'); run.define_metric('train/*', step_metric='update')
for update in sorted(rows): run.log({'update': update, **rows[update]})
run.finish()
print(json.dumps(dict(backfilled_updates=len(rows), run=run.url)))
