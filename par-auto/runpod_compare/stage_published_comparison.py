"""Verify published VPD and fetch the completed par-auto run on a CPU allocation."""
import base64
import hashlib
import json
from pathlib import Path
import torch
import wandb
from common import ROOT, R2_PREFIX, TARGET_SHA, credentials, sha, checkpoint, save_json, target
from stage_milestone import download


def main():
    credentials(); torch.set_num_threads(4)
    out = ROOT/'checkpoints/published_vs_finished'
    out.mkdir(parents=True, exist_ok=True)
    original = Path('/projects/RohanKathuria/param-clustering/data/vpd/model_400000.pth')
    public_file = wandb.Api().run('goodfire/spd/s-55ea3f9b').file('model_400000.pth')
    digest = hashlib.md5()
    with original.open('rb') as source:
        for block in iter(lambda: source.read(8<<20), b''): digest.update(block)
    assert public_file.md5 in (digest.hexdigest(), base64.b64encode(digest.digest()).decode())
    published = torch.load(original, map_location='cpu', weights_only=False, mmap=True)
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    handle_deprecated_state_dict_keys_(published)
    frozen = target(device='cpu')
    for name, value in frozen.state_dict().items():
        torch.testing.assert_close(published['target_model.'+name], value, rtol=0, atol=0)
    del frozen
    checkpoint(dict(method='vpd', completed=400000, training_positions=400000*32768,
                    target_sha256=TARGET_SHA, data_sha256=None, factors=published,
                    provenance=dict(kind='published checkpoint', run='goodfire/spd/s-55ea3f9b',
                                    file=original.name, sha256=sha(original), public_md5=public_file.md5)),
               out/'vpd.pth')
    download(R2_PREFIX+'/output/train/parauto/final.pth', out/'parauto_original.pth')
    download(R2_PREFIX+'/output/train/parauto/result.json', out/'parauto_result.json')
    info = json.loads((out/'parauto_result.json').read_text())
    saved = torch.load(out/'parauto_original.pth', map_location='cpu', weights_only=False, mmap=True)
    assert info['complete'] and info['positions'] == 1000000000
    assert info['completed'] == saved['completed'] == 30518
    assert saved['target_sha256'] == info['target_sha256'] == TARGET_SHA
    saved.update(training_positions=info['positions'], data_sha256=info['data_sha256'],
                 provenance=dict(kind='completed original training run', wandb_run='fhx2vfdm',
                                 original_sha256=sha(out/'parauto_original.pth')))
    checkpoint(saved, out/'parauto.pth')
    save_json({name:dict(path=str(out/(name+'.pth')), sha256=sha(out/(name+'.pth')))
               for name in ('vpd','parauto')}, out/'inputs_ready.json')
    print('PUBLISHED_COMPARISON_INPUTS_READY', flush=True)


if __name__ == '__main__': main()
