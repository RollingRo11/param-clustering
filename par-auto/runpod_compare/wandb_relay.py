"""Forward a run's metrics.jsonl to W&B from a machine with internet (KISSKI's compute nodes have none).

The training job (matryoshka13b.py rank 0) appends one JSON object per logged row: {"update": n, "<metric>": value, ...}.
This loop reads the lines added since the last pass (complete lines only) and logs each with step = update to one W&B
run, created on the first line and resumed after a restart (its id and the file offset are kept in --state). Rows with
the same update merge into one W&B step; rows below W&B's current step (a re-run after a crash) are dropped by W&B.
Stops after the training's result.json appears and every line has been sent.
Usage: wandb_relay.py --run-dir OUT --project matryoshka --name NAME [--entity E] [--interval 60]
"""
import argparse
import json
from pathlib import Path
import time

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True); ap.add_argument('--project', required=True)
    ap.add_argument('--name', required=True); ap.add_argument('--entity', default='rohan-kathuria-neu')
    ap.add_argument('--group', default='matryoshka-13b'); ap.add_argument('--interval', type=float, default=60)
    args = ap.parse_args()
    metrics, state_path = args.run_dir/'metrics.jsonl', args.run_dir/'wandb_relay.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(offset=0, id=None)
    run = None
    while True:
        if run is None and metrics.exists() and metrics.stat().st_size > state['offset']:
            plan = json.loads((args.run_dir/'plan.json').read_text())
            run = wandb.init(project=args.project, entity=args.entity, name=args.name, group=args.group, job_type='training',
                             config=plan, id=state['id'], resume='allow', dir=str(args.run_dir))
            state['id'] = run.id
            state_path.write_text(json.dumps(dict(state, url=run.url)))
            print(json.dumps(dict(relaying=str(metrics), wandb=run.url)), flush=True)
        sent = 0
        if run is not None:
            if metrics.stat().st_size < state['offset']: state['offset'] = 0         # the file was restarted
            with open(metrics, 'rb') as f:
                f.seek(state['offset'])
                chunk = f.read()
            end = chunk.rfind(b'\n') + 1                                           # complete lines only
            for line in chunk[:end].decode().splitlines():
                if not line.strip(): continue
                row = json.loads(line)
                step = int(row.pop('update'))
                run.log(row, step=step)
                sent += 1
            state['offset'] += end
            state_path.write_text(json.dumps(dict(state, url=run.url)))
        if sent: print(json.dumps(dict(time=time.strftime('%H:%M:%S'), rows=sent, offset=state['offset'])), flush=True)
        if (args.run_dir/'result.json').exists() and run is not None and metrics.stat().st_size == state['offset']:
            run.summary.update(json.loads((args.run_dir/'result.json').read_text())); run.finish()
            print('RELAY_DONE', flush=True)
            return
        time.sleep(args.interval)


if __name__ == '__main__': main()
