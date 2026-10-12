"""Bounded local monitor and evaluation submitter for the four mask-free trials.

Training continuation belongs to the immutable Slurm launcher. This process only
reads small metadata and submits missing evaluations. It never cancels jobs,
changes training sources, or loads tensors on a login node. Stops after 48 hours.
"""
import datetime as dt
import json
from pathlib import Path
import shlex
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'objective_train67_20260915_v3'
OUT = PROJECT / 'par-auto/out' / EXPERIMENT
REMOTE_PROJECT = '/projects/RohanKathuria/param-clustering'
ROOT = REMOTE_PROJECT + '/data/parauto/paired_1b_2000_20260910'
REMOTE = ROOT + '/output/' + EXPERIMENT
VARIANTS = ['finite_gim', 'finite_fisher', 'write', 'pair_gim']
PREP = dict(zip(VARIANTS, [10364948, 10365004, 10365005, 10365007]))

INVENTORY = r'''
import json, pathlib, re, subprocess, sys
root = pathlib.Path(sys.argv[1]); files = {}; ids = set(sys.argv[2:])
patterns = ['*/plan.json', '*/prepare.json', '*/progress.json', '*/latest.json',
 '*/wandb.json', '*/training_complete.json', '*/*submission.txt',
 '*/continuations.txt', '*/eval_*/complete.json', '*/eval_*/*/klkeep.json',
 'comparison.json', 'report.md']
for pattern in patterns:
 for path in root.glob(pattern):
  if path.is_file() and path.stat().st_size < 4000000:
   value = path.read_text(); files[str(path.relative_to(root))] = value
   if path.name.endswith('submission.txt') or path.name == 'continuations.txt':
    ids.update(re.findall(r'\b\d{7,}\b', value))
queue = subprocess.run(['squeue', '-u', 'kathuria.r', '-h', '-o', '%i|%j|%T|%M|%S|%R'],
 capture_output=True, text=True, check=True).stdout.strip().splitlines()
accounting = subprocess.run(['sacct', '-X', '-j', ','.join(sorted(ids)), '-n', '-P',
 '--format=JobID,State,ExitCode'], capture_output=True, text=True, check=True).stdout.strip().splitlines()
print(json.dumps(dict(files=files, queue=queue, accounting=accounting)))
'''


def remote(command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           'explorer', command], capture_output=True, text=True, timeout=55)


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def elapsed_seconds(value):
    days, sep, clock = value.partition('-')
    if not sep:
        days, clock = '0', days
    parts = [int(v) for v in clock.split(':')]
    return int(days) * 86400 + sum(v * 60**i for i, v in enumerate(reversed(parts)))


def submit(variant, step, repair_job=None):
    marker = REMOTE + f'/{variant}/eval_{step}_submission.txt'
    fixed_marker = REMOTE + f'/{variant}/eval_{step}_auxfix_submission.txt'
    launcher = ROOT + '/code/objective_train67_evalfix_20260915/slurm_objective_train67_eval_fix.sh'
    # Only the watcher submits baselines. Trial evaluations are submitted here
    # well before an active allocation's deadline, or after its launcher exits.
    condition = ('test "$(cat ' + shlex.quote(marker) + ')" = ' + shlex.quote(repair_job)
                 if repair_job else '! test -s ' + shlex.quote(marker))
    cmd = ('cd ' + shlex.quote(REMOTE_PROJECT) + ' && mkdir -p ' +
           shlex.quote(REMOTE + '/' + variant) + ' && if ' + condition +
           '; then job=$(sbatch --parsable --nice=100 --job-name=' +
           shlex.quote(f'{variant}-eval{step}') + ' ' + shlex.quote(launcher) + ' ' +
           shlex.quote(variant) + ' ' + str(step) + ') && printf \'%s\\n\' "$job" > ' +
           shlex.quote(marker) + ' && printf \'%s\\n\' "$job" > ' +
           shlex.quote(fixed_marker) + ' && printf \'%s\\n\' "$job"; fi')
    result = remote(cmd)
    return dict(variant=variant, step=step, returncode=result.returncode,
                output=result.stdout.strip(), error=result.stderr.strip())


def iteration():
    result = remote('python3 -c ' + shlex.quote(INVENTORY) + ' ' +
                    shlex.quote(REMOTE) + ' ' + ' '.join(str(j) for j in PREP.values()))
    result.check_returncode()
    data = json.loads(result.stdout)
    files = data.pop('files')
    for name, value in files.items():
        atomic(OUT / name, value)
    rows = [r.split('|') for r in data['queue']]
    states = {r.split('|')[0]: r.split('|')[1] for r in data['accounting']}
    failed = {j: s for j, s in states.items() if any(x in s for x in
              ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'])}
    prepared = all(json.loads(files.get(v + '/prepare.json', '{}')).get('passed') for v in VARIANTS)
    queue_by_id = {r[0]: r for r in rows}
    submitted = []
    # Reserve slots for the four training chains and the launcher hand-offs.
    slots = max(0, 10 - len(rows)) if prepared else 0
    # Step 250 selects the early 16-context split; these reference models still
    # retain their actual 5k/400k training steps inside the evaluator.
    candidates = [(v, step) for step in [250, 5000]
                  for v in ['gim_baseline', 'vpd_published']]
    candidates += [(v, step) for step in [250, 1000, 5000] for v in VARIANTS]
    for variant, step in candidates:
        if not slots:
            break
        prefix = variant + '/'
        previous = files.get(prefix + f'eval_{step}_submission.txt', '').strip()
        repair = (previous and states.get(previous) == 'FAILED' and
                  variant in ['finite_gim', 'finite_fisher'] and
                  prefix + f'eval_{step}/fisher/klkeep.json' in files and
                  prefix + f'eval_{step}/gim/klkeep.json' in files and
                  prefix + f'eval_{step}_auxfix_submission.txt' not in files)
        if previous and not repair:
            continue
        if prefix + f'eval_{step}/complete.json' in files:
            continue
        if variant in VARIANTS:
            latest = json.loads(files.get(prefix + 'latest.json', '{}')).get('step', 0)
            if latest < step:
                continue
            ids = []
            first = files.get(prefix + 'train_submission.txt', '').strip().split(';')[0]
            if first:
                ids.append(first)
            for line in files.get(prefix + 'continuations.txt', '').splitlines():
                ids.extend(line.split()[:2])
            active = [queue_by_id[j] for j in ids if j in queue_by_id]
            # Avoid racing the Slurm launcher's own evaluation submission.
            if not repair and active and not any(r[2] == 'RUNNING' and
                                  60 < elapsed_seconds(r[3]) < 5100 for r in active):
                continue
            if not active and any(states.get(j) not in ['COMPLETED', 'FAILED', 'TIMEOUT',
                                                       'OUT_OF_MEMORY', 'CANCELLED', 'NODE_FAIL'] for j in ids):
                continue
        receipt = submit(variant, step, previous if repair else None)
        submitted.append(receipt)
        if receipt['returncode'] == 0:
            slots -= 1
        else:
            break
    complete = all(v + '/eval_5000/complete.json' in files for v in
                   VARIANTS + ['gim_baseline', 'vpd_published'])
    progress = {v: json.loads(files.get(v + '/progress.json', '{}')).get('update', 0) for v in VARIANTS}
    state = dict(status='complete' if complete else 'needs_attention' if failed else 'waiting',
                 checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                 preparation_passed=prepared, updates=progress, failed=failed,
                 submitted=submitted, **data)
    atomic(OUT / 'watch_status.json', json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)
    return complete


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 48 * 3600
    while time.monotonic() < deadline:
        try:
            if iteration():
                return
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            print(json.dumps(dict(status='connection_retry', error=str(exc))), flush=True)
        time.sleep(180)
    atomic(OUT / 'watch_expired.json', json.dumps(dict(
        checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        note='48-hour monitor limit reached. Cluster jobs were left unchanged.')) + '\n')


if __name__ == '__main__':
    main()
