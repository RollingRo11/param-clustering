"""Continuously upload atomic checkpoint versions; never copy credentials."""
import argparse
import json
import os
from pathlib import Path
import time

from common import ROOT, credentials, save_json, upload


def sync_once(state):
    files = list((ROOT/'output').rglob('*')) + list((ROOT/'logs').glob('*.log'))
    for path in sorted(files):
        relative = str(path.relative_to(ROOT))
        if not path.is_file() or path.is_symlink() or 'wandb' in path.parts: continue
        if path.name in ('backup_state.json', 'backup.log') or path.suffix not in ('.json', '.pt', '.pth', '.log', '.png', '.csv'): continue
        stat = path.stat(); version = [stat.st_mtime_ns, stat.st_size]
        if state.get(relative, {}).get('version') == version: continue
        # Checkpoints are written by atomic rename. A hard link retains a
        # stable inode while the trainer replaces checkpoint.pt again.
        frozen = ROOT/'upload-staging'/path.name
        frozen.parent.mkdir(exist_ok=True)
        frozen.unlink(missing_ok=True)
        if path.suffix in ('.pt', '.pth'): os.link(path, frozen)
        else: frozen.write_bytes(path.read_bytes())
        try:
            receipt = upload(frozen, relative)
            state[relative] = dict(version=version, **receipt)
            save_json(state, ROOT/'output/backup_state.json')
            print(json.dumps(dict(uploaded=relative, bytes=receipt['bytes'])), flush=True)
        finally: frozen.unlink(missing_ok=True)
    return state


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--once', action='store_true'); args = parser.parse_args()
    credentials()
    path = ROOT/'output/backup_state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    while True:
        try:
            sync_once(state)
            upload(path, 'output/backup_state.json')
            if args.once: return
        except Exception as exc:
            print(f'Backup retry: {type(exc).__name__}: {exc}', flush=True)
            if args.once: raise
        time.sleep(30)


if __name__ == '__main__': main()
