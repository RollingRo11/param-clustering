"""Fetch permanent factor snapshots without replaying or touching live trainers."""
import argparse
import json
import torch
from common import ROOT, R2_PREFIX, TARGET_SHA, credentials, sha, checkpoint, save_json
from stage_milestone import download


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--step',type=int,required=True)
    args=parser.parse_args(); credentials(); torch.set_num_threads(4)
    out=ROOT/'checkpoints'/f'step_{args.step}'; out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    receipts={}
    for name in ('parauto','vpd'):
        source=out/f'{name}_original.pth'
        key=f'{R2_PREFIX}/output/train/{name}/model_{args.step}.pth'
        download(key,source)
        saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
        assert saved['step']==args.step and saved['target_sha256']==TARGET_SHA
        destination=out/f'{name}.pth'
        checkpoint(dict(method=name,completed=args.step,training_positions=args.step*32768,
                        data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA,
                        factors=saved['factors'],provenance=dict(kind='original retained factor snapshot',
                        source_key=key,original_sha256=sha(source))),destination)
        receipts[name]=dict(path=str(destination),sha256=sha(destination),step=args.step,source_key=key)
    save_json(receipts,out/'inputs_ready.json')
    print('RETAINED_SNAPSHOT_INPUTS_READY',flush=True)


if __name__=='__main__': main()
