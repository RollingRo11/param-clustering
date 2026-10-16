"""Archive a directory to R2 before freeing space on the cluster. Every file under SRC (except the excluded top-level
names; symlinks skipped) is uploaded to KEY_PREFIX/<path relative to SRC> with its sha256 in the object metadata (as
common.upload does), then every object is downloaded again and hashed. Writes OUT/manifest.jsonl (one row per file:
path, key, bytes, sha256, status, verified) and OUT/verified_delete_candidates.txt (absolute paths of files with one of
--suffixes whose R2 copy hashed equal and which did not change during the run), and uploads the manifest next to the
archive. It never deletes anything and never replaces an existing object whose content differs (status conflict).
Re-running skips objects already uploaded with the same sha256.
Usage: r2_archive.py SRC KEY_PREFIX OUT [--exclude NAME ...] [--suffixes .pt .pth ...] [--workers 32]"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

BUCKET = 'param-clustering'
BLOCK = 8 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(BLOCK), b''): h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path); ap.add_argument('prefix'); ap.add_argument('out', type=Path)
    ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--suffixes', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip('/')
    client = boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT'], aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
                          aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto',
                          config=Config(max_pool_connections=4 * args.workers, retries=dict(max_attempts=10, mode='adaptive')))
    transfer = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=4)
    files = []
    for top in sorted(args.src.iterdir()):
        if top.name in args.exclude or top.is_symlink(): continue
        if top.is_file(): files.append(top)
        elif top.is_dir():
            for root, _, names in os.walk(top):          # does not follow symlinked directories
                files += [Path(root)/n for n in names if (Path(root)/n).is_file() and not (Path(root)/n).is_symlink()]
    files.sort()
    existing = {}
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=prefix + '/'):
        for o in page.get('Contents', []): existing[o['Key']] = o['Size']
    print(json.dumps(dict(files=len(files), gb=round(sum(f.stat().st_size for f in files) / 1e9, 2),
                          existing_objects=len(existing), prefix=prefix, excluded=args.exclude)), flush=True)
    lock, progress, started = threading.Lock(), dict(files=0, bytes=0), time.time()

    def tick(stage, size):
        with lock:
            progress['files'] += 1; progress['bytes'] += size
            if progress['files'] % 2000 == 0 or size > 1 << 30:
                print(json.dumps(dict(stage=stage, files=progress['files'], gb=round(progress['bytes'] / 1e9, 1),
                                      minutes=round((time.time() - started) / 60, 1))), flush=True)

    def put(path):
        stat = path.stat()
        row = dict(path=str(path), key=f'{prefix}/{path.relative_to(args.src).as_posix()}', bytes=stat.st_size,
                   mtime_ns=stat.st_mtime_ns)
        try:
            row['sha256'] = sha256_file(path)
            if row['key'] in existing:
                info = client.head_object(Bucket=BUCKET, Key=row['key'])
                same = info['ContentLength'] == row['bytes'] and info.get('Metadata', {}).get('sha256') == row['sha256']
                row['status'] = 'already' if same else 'conflict'
            else:
                client.upload_file(str(path), BUCKET, row['key'], ExtraArgs={'Metadata': {'sha256': row['sha256']}}, Config=transfer)
                info = client.head_object(Bucket=BUCKET, Key=row['key'])
                ok = info['ContentLength'] == row['bytes'] and info['Metadata'].get('sha256') == row['sha256']
                row['status'] = 'uploaded' if ok else 'upload_mismatch'
        except Exception as exc:
            row['status'], row['error'] = 'error', f'{type(exc).__name__}: {exc}'
            with lock:
                progress['errors'] = progress.get('errors', 0) + 1
                if progress['errors'] <= 5: print(json.dumps(dict(upload_error=row['path'], error=row['error'])), flush=True)
        tick('upload', row['bytes'])
        return row

    def check(row):
        """Download the object and hash it; the local file must also be unchanged since it was hashed."""
        row['verified'] = False
        if row['status'] in ('uploaded', 'already'):
            try:
                h, size = hashlib.sha256(), 0
                for block in client.get_object(Bucket=BUCKET, Key=row['key'])['Body'].iter_chunks(BLOCK):
                    h.update(block); size += len(block)
                stat = Path(row['path']).stat()
                row['verified'] = (h.hexdigest() == row['sha256'] and size == row['bytes'] == stat.st_size
                                   and stat.st_mtime_ns == row['mtime_ns'])
            except Exception as exc:
                row['verify_error'] = f'{type(exc).__name__}: {exc}'
        tick('verify', row['bytes'])
        return row

    with ThreadPoolExecutor(args.workers) as pool: rows = list(pool.map(put, files))
    progress.update(files=0, bytes=0)
    with ThreadPoolExecutor(args.workers) as pool: rows = list(pool.map(check, rows))
    manifest = args.out/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    candidates = [r for r in rows if r['verified'] and r['path'].endswith(tuple(args.suffixes))]
    (args.out/'verified_delete_candidates.txt').write_text(''.join(r['path'] + '\n' for r in candidates))
    client.upload_file(str(manifest), BUCKET, f'{prefix}/_archive_manifest_{time.strftime("%Y%m%d_%H%M%S")}.jsonl')
    status = {}
    for r in rows: status[r['status']] = status.get(r['status'], 0) + 1
    problems = [dict(path=r['path'], status=r['status'], error=r.get('error') or r.get('verify_error'))
                for r in rows if not r['verified']]
    print(json.dumps(dict(status=status, verified_files=sum(r['verified'] for r in rows),
                          verified_gb=round(sum(r['bytes'] for r in rows if r['verified']) / 1e9, 2),
                          not_verified=len(problems), first_problems=problems[:20],
                          delete_candidates=len(candidates), delete_candidates_gb=round(sum(r['bytes'] for r in candidates) / 1e9, 2),
                          minutes=round((time.time() - started) / 60, 1))), flush=True)
    print('R2_ARCHIVE_DONE', flush=True)


if __name__ == '__main__': main()
