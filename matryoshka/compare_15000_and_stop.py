"""Recover/evaluate 15k checkpoints, then save training and release the pod.

Takes over lifecycle management from supervise.py without interrupting VPD
until the comparison finishes. Shutdown is gated on verified R2 backups.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import requests
from common import ROOT, HERE, credentials, save_json, upload, sha, s3

OUTPUT = ROOT/'output/evaluations/15000'
STATUS = OUTPUT/'job_status.json'


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1][0] != 'Z'


def check_command(pid, expected):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if expected not in command:
        raise RuntimeError(f'Process {pid} is not {expected}')


def main():
    credentials(); OUTPUT.mkdir(parents=True, exist_ok=True)
    # A lock prevents two lifecycle controllers from running this sequence.
    import fcntl
    lock = open(OUTPUT/'job.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = json.loads((ROOT/'output/supervisor.json').read_text())
    supervisor = existing['pid']; trainer = existing['pids']['vpd']
    uploader_ids = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes().replace(b'\0', b' ').decode()
            if str(HERE/'backup.py') in command and command.startswith(sys.executable):
                uploader_ids.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    check_command(supervisor, 'supervise.py'); check_command(trainer, 'train_pair.py vpd')
    state = dict(pid=os.getpid(), started=time.time(), step=15000, vpd_pid=trainer,
                 prior_supervisor=existing, backup_pids=uploader_ids,
                 stop_after_comparison=True, phase='taking_over')
    def status(phase, **values):
        state.update(phase=phase, updated=time.time(), **values)
        save_json(state, STATUS)
        print(json.dumps(dict(phase=phase, **values)), flush=True)
    status('taking_over')
    # supervise.py has no SIGTERM handler; its child trainers remain alive.
    os.kill(supervisor, signal.SIGTERM)
    while alive(supervisor): time.sleep(1)
    save_json(dict(pid=os.getpid(), phase='15k_comparison_then_stop',
                   status_file=str(STATUS), pids={'vpd': trainer}), ROOT/'output/supervisor.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
    def run(script, *args):
        subprocess.run([sys.executable, '-u', str(HERE/script), *map(str, args)],
                       cwd=HERE.parent/'vpd', env=env, check=True)
    error = None
    try:
        # Keep original training source provenance intact; archive this addition
        # separately, including its shutdown controller and evaluator.
        source = OUTPUT/'source.tar.gz'
        paths = [p for p in HERE.iterdir() if p.suffix in ('.py', '.yaml', '.md', '.sh')]
        save_json({p.name: sha(p) for p in paths}, OUTPUT/'source_manifest.json')
        with tarfile.open(source, 'w:gz') as tar:
            for path in paths: tar.add(path, arcname=path.name, recursive=False)
        save_json(upload(source, 'evaluations/15000/source.tar.gz'), OUTPUT/'source_r2.json')
        status('recovering_parauto')
        run('replay_parauto.py', '--step', 15000)
        for method in ('parauto', 'vpd'):
            milestone = ROOT/'output/milestones/15000'/method
            receipt = json.loads((milestone/'r2_ready.json').read_text())
            assert receipt['completed'] == 15000 and receipt['method'] == method
            status('evaluating_'+method)
            run('evaluate_checkpoint.py', method, '--checkpoint', milestone/'model.pth',
                '--output', OUTPUT/method, '--expected-step', 15000)
        status('plotting')
        run('report_milestone.py', '--step', 15000, '--output', OUTPUT)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    status('saving_final_training_state', comparison_succeeded=error is None, error=error)
    if alive(trainer):
        check_command(trainer, 'train_pair.py vpd')
        os.kill(trainer, signal.SIGUSR1)
        while alive(trainer): time.sleep(5)
    final = json.loads((ROOT/'output/train/vpd/result.json').read_text())
    marker = json.loads((ROOT/'output/train/vpd/checkpoint_ready.json').read_text())
    assert final['completed'] == marker['completed'] >= 15000
    for pid in uploader_ids:
        if alive(pid):
            check_command(pid, 'backup.py'); os.kill(pid, signal.SIGTERM)
            while alive(pid): time.sleep(1)
    status('verifying_R2', final_vpd_step=final['completed'])
    # Network failures retry indefinitely; never discard unverified artifacts.
    while True:
        try:
            run('backup.py', '--once')
            client = s3()
            receipts = json.loads((ROOT/'output/backup_state.json').read_text())
            for receipt in receipts.values():
                head = client.head_object(Bucket='param-clustering', Key=receipt['key'])
                assert head['ContentLength'] == receipt['bytes']
                assert head['Metadata']['sha256'] == receipt['sha256']
            # Also recheck the shared dataset and original source archive.
            def verify(value):
                if isinstance(value, dict):
                    if all(k in value for k in ('key', 'bytes', 'sha256')):
                        head = client.head_object(Bucket='param-clustering', Key=value['key'])
                        assert head['ContentLength'] == value['bytes']
                        assert head['Metadata']['sha256'] == value['sha256']
                    else:
                        for child in value.values(): verify(child)
                elif isinstance(value, list):
                    for child in value: verify(child)
            for name in ('data_r2.json', 'source_r2.json'):
                verify(json.loads((ROOT/'output'/name).read_text()))
            # Include the exact target under this experiment's own prefix.
            upload(ROOT/'data/model_step_99999.pt', 'data/model_step_99999.pt')
            status('stop_requested', r2_verified=True, verified_output_files=len(receipts))
            upload(STATUS, 'evaluations/15000/job_status.json')
            upload(STATUS, 'output/evaluations/15000/job_status.json')
            upload(ROOT/'logs/compare-15000.log', 'logs/compare-15000.log')
            response = requests.post('https://rest.runpod.io/v1/pods/sz9usox6q9hihe/stop',
                                     headers={'Authorization': 'Bearer '+os.environ['RUNPOD_API_KEY']}, timeout=60)
            response.raise_for_status()
            return
        except Exception:
            print(traceback.format_exc(), flush=True)
            time.sleep(30)


if __name__ == '__main__': main()
