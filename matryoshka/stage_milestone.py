"""Pull the shared eval assets and exact milestone exports from R2 on a CPU allocation."""
import argparse
import json
import os
import tarfile
import time
from pathlib import Path
from botocore.exceptions import ClientError
from common import ROOT, R2_PREFIX, TARGET_SHA, credentials, s3, sha, save_json


def download(key, path, digest=None):
    client = s3(); path = Path(path)
    head = client.head_object(Bucket='param-clustering', Key=key)
    expected = digest or head.get('Metadata', {}).get('sha256')
    if not expected: raise ValueError(f'Missing digest: {key}')
    if path.exists() and sha(path) == expected: return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.download')
    client.download_file('param-clustering', key, str(temp))
    if sha(temp) != expected or temp.stat().st_size != head['ContentLength']:
        raise ValueError(f'Download verification failed: {key}')
    os.replace(temp, path)
    print(json.dumps(dict(downloaded=key, bytes=head['ContentLength'])), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--setup', action='store_true')
    parser.add_argument('--step', type=int, default=2000); parser.add_argument('--hours', type=float, default=3)
    args = parser.parse_args(); credentials()
    if args.setup:
        archive = ROOT/'source.tar.gz'
        download(R2_PREFIX+'/source.tar.gz', archive)
        with tarfile.open(archive) as tar: tar.extractall(ROOT/'code', filter='data')
        download('cofac67/target/model_step_99999.pt', ROOT/'data/model_step_99999.pt', TARGET_SHA)
        for filename in ('manifest.json', 'val.uint16', 'val.strata.json', 'val.permutation.npy'):
            download(R2_PREFIX+'/data/'+filename, ROOT/'data'/filename)
        return
    deadline = time.monotonic()+args.hours*3600
    done = {}
    while len(done) != 2:
        for name in ('parauto', 'vpd'):
            if name in done: continue
            key = f'{R2_PREFIX}/milestones/{args.step}/{name}/r2_ready.json'
            try: info = json.loads(s3().get_object(Bucket='param-clustering', Key=key)['Body'].read())
            except ClientError as exc:
                if exc.response['Error']['Code'] in ('NoSuchKey', '404'): continue
                raise
            if info['completed'] != args.step or info['method'] != name or info['target_sha256'] != TARGET_SHA:
                raise ValueError('Milestone manifest mismatch')
            download(info['model']['key'], ROOT/'checkpoints'/name/'model.pth', info['model']['sha256'])
            done[name] = info
        if time.monotonic() > deadline: raise TimeoutError('Milestone exports not ready before staging deadline')
        if len(done) != 2: time.sleep(30)
    if done['parauto']['training_data_sha256'] != done['vpd']['training_data_sha256']:
        raise ValueError('Training data mismatch')
    save_json(done, ROOT/'inputs_ready.json')
    print('MILESTONE_INPUTS_READY', flush=True)


if __name__ == '__main__': main()
