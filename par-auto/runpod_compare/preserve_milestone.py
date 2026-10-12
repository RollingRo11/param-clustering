"""Preserve exact checkpoints while trainers continue, then export to R2."""
import argparse
import json
import os
import time
import torch
from botocore.exceptions import BotoCoreError, ClientError
from common import ROOT, TARGET_SHA, checkpoint, credentials, save_json, upload


def preserve(name, step):
    source = ROOT/'output/train'/name
    destination = ROOT/'output/milestones'/str(step)/name
    destination.mkdir(parents=True, exist_ok=True)
    ready = destination/'r2_ready.json'
    if ready.exists():
        upload(ready, f'milestones/{step}/{name}/r2_ready.json')
        return True
    pinned = destination/'checkpoint.pt'
    if not pinned.exists():
        marker = source/'checkpoint_ready.json'
        if not marker.exists(): return False
        completed = json.loads(marker.read_text())['completed']
        if completed < step: return False
        if completed > step: raise RuntimeError(f'{name}: requested checkpoint {step} was missed; latest is {completed}')
        os.link(source/'checkpoint.pt', pinned)
    saved = torch.load(pinned, map_location='cpu', weights_only=False, mmap=True)
    if saved['completed'] != step or saved['method'] != name or saved['target_sha256'] != TARGET_SHA:
        raise ValueError(f'{name}: checkpoint identity mismatch')
    factors = saved['state']['pieces'] if name == 'parauto' else saved['state']['model']
    export = destination/'model.pth'
    if not export.exists():
        checkpoint(dict(completed=step, method=name, target_sha256=TARGET_SHA,
                        data_sha256=saved['data_sha256'], factors=factors), export)
    info = dict(method=name, completed=step, positions=step*32768,
                training_data_sha256=saved['data_sha256'], target_sha256=TARGET_SHA,
                model=upload(export, f'milestones/{step}/{name}/model.pth'))
    save_json(info, ready)
    upload(ready, f'milestones/{step}/{name}/r2_ready.json')
    print(json.dumps(info), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--step', type=int, default=2000)
    args = parser.parse_args(); credentials()
    finished = set()
    while len(finished) < 2:
        for name in ('parauto', 'vpd'):
            if name in finished: continue
            try:
                if preserve(name, args.step): finished.add(name)
            except (BotoCoreError, ClientError, OSError) as exc:
                print(f'{name}: retrying milestone export after {type(exc).__name__}', flush=True)
        if len(finished) == 2: return
        time.sleep(15)


if __name__ == '__main__': main()
