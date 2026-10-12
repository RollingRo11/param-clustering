"""Launch the matched pair, evaluate, back up, and release both GPUs."""
import json
import os
import signal
import subprocess
import sys
import time
import requests

from common import ROOT, HERE, POSITIONS, STEPS, credentials, save_json, upload


def launch(method, script, extra=()):
    gpu = '0' if method == 'vpd' else '1'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    log = open(ROOT/'logs'/f'{script}-{method}.log', 'a')
    child = subprocess.Popen([sys.executable, '-u', str(HERE/(script+'.py')), method, *extra],
                              cwd=HERE.parent/'vpd', env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return child


def main():
    credentials()
    while not (ROOT/'output/data_r2.json').exists():
        save_json(dict(pid=os.getpid(), phase='waiting_for_verified_dataset'), ROOT/'output/supervisor.json')
        time.sleep(15)
    for name in ('vpd', 'parauto'):
        pilot = json.loads((ROOT/'output/pilot'/name/'result.json').read_text())
        if pilot['completed'] != 20: raise ValueError('Expected successful 20-update pilot')
        if not (ROOT/'output/pilot'/name/'klkeep.json').exists(): raise ValueError('Evaluation smoke missing')
    start = time.time()
    pid = ROOT/'output/supervisor.json'
    save_json(dict(pid=os.getpid(), started=start, phase='training', positions_per_method=POSITIONS), pid)
    backup_log = open(ROOT/'logs/backup.log', 'a')
    uploader = subprocess.Popen([sys.executable, '-u', str(HERE/'backup.py')], stdout=backup_log, stderr=subprocess.STDOUT)
    jobs = {name: launch(name, 'train_pair', ['--resume'] if (ROOT/'output/train'/name/'checkpoint.pt').exists() else [])
            for name in ('vpd', 'parauto')}
    failed = False
    while any(p.poll() is None for p in jobs.values()):
        limits_path = ROOT/'limits.json'
        limits = json.loads(limits_path.read_text()) if limits_path.exists() else {}
        exceeded = limits.get('max_hours') is not None and time.time()-start > limits['max_hours']*3600
        if exceeded or any(p.poll() not in (0, None) for p in jobs.values()):
            failed = True
            for p in jobs.values():
                if p.poll() is None: p.send_signal(signal.SIGUSR1)
        save_json(dict(pid=os.getpid(), started=start, phase='training', failed=failed,
                       pids={n: p.pid for n, p in jobs.items()}, exits={n: p.poll() for n, p in jobs.items()}), pid)
        time.sleep(15)
    if not failed:
        for name in jobs:
            result = json.loads((ROOT/'output/train'/name/'result.json').read_text())
            if not result['complete'] or result['positions'] != POSITIONS or result['completed'] != STEPS:
                failed = True
    if not failed:
        jobs = {name: launch(name, 'evaluate') for name in jobs}
        save_json(dict(pid=os.getpid(), started=start, phase='evaluation', pids={n: p.pid for n, p in jobs.items()}), pid)
        for p in jobs.values(): failed |= p.wait() != 0
        if not failed:
            failed = subprocess.run([sys.executable, str(HERE/'report.py')]).returncode != 0
    # Stop the background uploader cleanly between files before the final
    # pass. It has no training state; a killed multipart upload is retried.
    uploader.terminate(); uploader.wait(); backup_log.close()
    save_json(dict(started=start, finished=time.time(), phase='complete' if not failed else 'stopped_with_error',
                   failed=failed), pid)
    # Never shut down until artifacts have been uploaded and HEAD verified.
    while True:
        result = subprocess.run([sys.executable, '-u', str(HERE/'backup.py'), '--once'])
        if result.returncode == 0: break
        time.sleep(30)
    upload(pid, 'output/supervisor.json')
    # Stop retains /workspace as an additional recoverable copy.
    response = requests.post('https://rest.runpod.io/v1/pods/sz9usox6q9hihe/stop',
                             headers={'Authorization': 'Bearer '+os.environ['RUNPOD_API_KEY']}, timeout=60)
    response.raise_for_status()


if __name__ == '__main__': main()
