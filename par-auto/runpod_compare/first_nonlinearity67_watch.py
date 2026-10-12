"""Bounded local status/copy-back helper for the submitted objective screen.

Reads scheduler/status files only. Does not submit, cancel, or modify any job.
Stops on completion, failure, or after four hours. Polls once per minute.
"""
import datetime as dt
import json
from pathlib import Path
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT/'par-auto/out/first_nonlinearity67_20260915'
REMOTE_PROJECT = '/projects/RohanKathuria/param-clustering'
REMOTE = REMOTE_PROJECT+'/data/parauto/paired_1b_2000_20260910/output/first_nonlinearity67_20260915'
GPU_JOB, ANALYSIS_JOB = 10364399, 10364399


def remote(command):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','explorer',command],
                          capture_output=True,text=True,timeout=40)


def fetch():
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',
                    'explorer:'+REMOTE+'/*.json','explorer:'+REMOTE+'/*.md',str(OUT)],check=True,timeout=120)


def save(state):
    state['checked_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    temp = OUT/'watch_status.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n')
    temp.replace(OUT/'watch_status.json')
    print(json.dumps(state),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+4*3600
    copied_main=False
    while time.monotonic()<deadline:
        try:
            done=remote(f'test -f {REMOTE}/analysis_complete.json')
            if done.returncode==0:
                fetch();save(dict(status='complete',copied_reports=True));return
            main_done=remote(f'test -f {REMOTE}/complete.json')
            if main_done.returncode==0 and not copied_main:
                fetch();copied_main=True
            status=remote(f'sacct -X -j {GPU_JOB},{ANALYSIS_JOB} --noheader --parsable2 --format=JobID,State,ExitCode')
            if status.returncode:
                save(dict(status='connection_retry',returncode=status.returncode))
            else:
                rows=[line.strip() for line in status.stdout.splitlines() if line.strip()]
                failed=any(any(word in line for word in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']) for line in rows)
                if failed:
                    subprocess.run(['scp','-q','-o','BatchMode=yes','explorer:'+REMOTE_PROJECT+f'/logs/first_nonlinearity67_{GPU_JOB}.log',str(OUT/'run.log')],timeout=40)
                    save(dict(status='needs_attention',jobs=rows,copied_main=copied_main));return
                save(dict(status='waiting',jobs=rows,copied_main=copied_main))
        except (OSError,subprocess.SubprocessError) as exc:
            save(dict(status='retry',error=str(exc)))
        time.sleep(60)
    save(dict(status='watch_expired',note='Jobs were left unchanged; check Explorer for later results.'))


if __name__=='__main__':main()
