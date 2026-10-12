"""Verify the original VPD 1k snapshot and the immutable GIM 1k export."""
import json
import torch
from common import ROOT, R2_PREFIX, TARGET_SHA, credentials, sha, checkpoint, save_json
from stage_milestone import download

def main():
    credentials(); torch.set_num_threads(4)
    out=ROOT/'checkpoints/gim_vpd_1000'; out.mkdir(parents=True,exist_ok=True)
    key=R2_PREFIX+'/output/train/vpd/model_1000.pth'
    source=out/'vpd_original.pth'
    download(key,source,'ef80d5b4999092fc3149ecbe32dd9b3d0f6f043df7dd7b76c5dace43645bbe58')
    saved=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==1000 and saved['target_sha256']==TARGET_SHA
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    checkpoint(dict(method='vpd',completed=1000,training_positions=32768000,
                    data_sha256=manifest['train']['sha256'],target_sha256=TARGET_SHA,
                    factors=saved['factors'],provenance=dict(kind='original retained snapshot',
                    source_key=key,original_sha256=sha(source))),out/'vpd.pth')
    del saved
    gim=ROOT/'output/gim_1h/checkpoints/001000/model.pth'
    saved=torch.load(gim,map_location='cpu',weights_only=False,mmap=True)
    assert saved['completed']==1000 and saved['sensor']=='gim' and saved['target_sha256']==TARGET_SHA
    assert saved['data_sha256']==manifest['train']['sha256']
    save_json(dict(vpd=dict(path=str(out/'vpd.pth'),sha256=sha(out/'vpd.pth')),
                   gim=dict(path=str(gim),sha256=sha(gim))),out/'inputs_ready.json')
    print('GIM_VPD_1000_INPUTS_READY',flush=True)

if __name__=='__main__': main()
