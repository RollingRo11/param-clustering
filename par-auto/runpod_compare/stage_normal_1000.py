"""Retrieve the original normal-gradient par-auto 1,000-step snapshot."""
import json
import torch
from common import ROOT, R2_PREFIX, TARGET_SHA, credentials, sha, checkpoint, save_json
from stage_milestone import download

def main():
    credentials(); torch.set_num_threads(4)
    out=ROOT/'checkpoints/normal_1000'; out.mkdir(parents=True,exist_ok=True)
    key=R2_PREFIX+'/output/train/parauto/model_1000.pth'
    source=out/'original.pth'
    download(key,source,'a005eb28a99cd2465f0cfc4d606782aab8e7da4307ce30ea209ab95e489e96ff')
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==1000 and saved['target_sha256']==TARGET_SHA
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    checkpoint(dict(method='parauto',completed=1000,training_positions=32768000,
        data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA,factors=saved['factors'],
        provenance=dict(kind='original retained normal-gradient snapshot',sensor='gradient',
        source_key=key,original_sha256=sha(source),schedule_horizon=30518)),out/'model.pth')
    save_json(dict(path=str(out/'model.pth'),sha256=sha(out/'model.pth'),source_key=key),out/'inputs_ready.json')
    print('NORMAL_1000_INPUTS_READY',flush=True)

if __name__=='__main__': main()
