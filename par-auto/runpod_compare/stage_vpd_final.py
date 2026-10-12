"""Fetch the paired VPD run's final state (and the 15k milestone) from R2 as evaluation checkpoints."""
import json, torch
from common import ROOT, R2_PREFIX, TARGET_SHA, credentials, sha, checkpoint, save_json
from stage_milestone import download


def stage(key, out_dir, expect_step=None):
    out_dir.mkdir(parents=True, exist_ok=True); source = out_dir/'vpd_original.pth'
    download(f'{R2_PREFIX}/{key}', source)
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    step = saved.get('step', saved.get('completed'))
    assert step is not None and saved['target_sha256'] == TARGET_SHA and (expect_step is None or step == expect_step), (key, step)
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    checkpoint(dict(method='vpd', completed=int(step), training_positions=int(step)*32768, data_sha256=manifest['train']['sha256'],
                    target_sha256=TARGET_SHA, factors=saved['factors'],
                    provenance=dict(kind='paired-run VPD checkpoint from R2', source_key=key, original_sha256=sha(source))), out_dir/'vpd.pth')
    save_json(dict(step=int(step), source_key=key, sha256=sha(out_dir/'vpd.pth')), out_dir/'inputs_ready.json')
    print(json.dumps(dict(staged=str(out_dir/'vpd.pth'), step=int(step))), flush=True)


if __name__ == '__main__':
    credentials(); torch.set_num_threads(2)
    stage('output/train/vpd/final.pth', ROOT/'checkpoints/vpd_final')
    stage('milestones/15000/vpd/model.pth', ROOT/'checkpoints/step_15000', 15000)
    print('STAGE_VPD_FINAL_DONE', flush=True)
