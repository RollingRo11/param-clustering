"""Recover a missing milestone by replaying the original saved optimizer state.

The full-run schedule is unchanged. Retained original factor snapshots are
used as independent checks before the recovered milestone is published.
"""
import argparse
from dataclasses import asdict
import json
import random
import time
import torch
import numpy as np
import wandb
from common import ROOT, MODULES, TARGET_SHA, STEPS, credentials, tokens, batch, target, checkpoint, save_json, sha, upload
from train_pair import capture, core, method


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--step', type=int, default=15000)
    args = parser.parse_args(); credentials()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    out = ROOT/'output/replay'/f'parauto_{args.step}'
    out.mkdir(parents=True, exist_ok=True)
    initial_path = ROOT/'output/milestones/2000/parauto/checkpoint.pt'
    recovery_path = out/'checkpoint.pt'
    source = recovery_path if recovery_path.exists() else initial_path
    saved = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    data, manifest = tokens()
    assert saved['method'] == 'parauto' and saved['target_sha256'] == TARGET_SHA
    assert saved['data_sha256'] == manifest['train']['sha256']
    state = saved['state']; start = state['next_step']
    assert 2000 <= start <= args.step < STEPS
    cfg = method.TrainConfig(**state['config'])
    assert cfg.steps == STEPS
    previous = json.loads((out/'wandb.json').read_text()) if recovery_path.exists() else None
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu',
                     name=f'paired-1b-parauto-replay-to-{args.step}', group='paired-h200-1b-20260910',
                     job_type='checkpoint-recovery', dir=str(out), id=previous['id'] if previous else None,
                     resume='must' if previous else None,
                     config=dict(training_config=asdict(cfg), initial_step=2000, stop_step=args.step,
                                 data_sha256=manifest['train']['sha256'], target_sha256=TARGET_SHA,
                                 purpose='Recover overwritten checkpoint using unchanged full-run schedule'))
    save_json(dict(id=run.id, url=run.url), out/'wandb.json')
    run.define_metric('update'); run.define_metric('replay/*', step_metric='update')
    model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in state['pieces']]
    # fit restores the saved normalizers. These tiny entries only supply its
    # matrix schema and have no effect on the restored objective or updates.
    sample = capture(model, batch(data, 0)[:1])
    reference = [method.slice_entry(e, slice(0, 256)) for e in sample]
    del sample
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
    verification_path = out/'verification.json'
    verifications = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    started = time.perf_counter()
    def log(row):
        if row['matrix_index'] == 0:
            run.log({'update': row['step']+1, 'replay/total_loss': row['total_loss']})
    def progress(completed):
        if completed % 100 == 0:
            torch.cuda.synchronize()
            print(json.dumps(dict(completed=completed, target=args.step,
                                  seconds_per_update=(time.perf_counter()-started)/(completed-start))), flush=True)
        return False
    def save(new_state):
        step = new_state['next_step']
        original_path = ROOT/'output/train/parauto'/f'model_{step}.pth'
        if original_path.exists():
            original = torch.load(original_path, map_location='cpu', weights_only=False, mmap=True)
            assert original['step'] == step and original['target_sha256'] == TARGET_SHA
            exact = True; maximum = 0.
            for actual, expected in zip(new_state['pieces'], original['factors'], strict=True):
                for name in ('read', 'write'):
                    a, b = actual[name].cpu(), expected[name]
                    exact &= torch.equal(a, b)
                    maximum = max(maximum, float((a-b).abs().max()))
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
            verifications[str(step)] = dict(bitwise_equal=exact, maximum_absolute_difference=maximum,
                                            original_sha256=sha(original_path))
            save_json(verifications, verification_path)
            print(json.dumps(dict(verified_step=step, **verifications[str(step)])), flush=True)
        checkpoint(dict(method='parauto', completed=step, data_sha256=manifest['train']['sha256'],
                        target_sha256=TARGET_SHA, state=new_state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state()), recovery_path)
    fitted, _ = method.fit(reference, weights, pieces, cfg, resume=state, log=log,
                            batch_provider=lambda step: capture(model, batch(data, step)),
                            stop_after=args.step-start, step_callback=progress,
                            checkpoint=save, checkpoint_interval=1000, allow_partial_batch=True)
    assert all(str(step) in verifications for step in (5000, 10000) if step <= args.step)
    destination = ROOT/'output/milestones'/str(args.step)/'parauto'
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination/'model.pth'
    provenance = dict(kind='replayed from original full checkpoint', from_step=2000, verified=verifications)
    checkpoint(dict(completed=args.step, method='parauto', target_sha256=TARGET_SHA,
                    data_sha256=manifest['train']['sha256'], provenance=provenance,
                    factors=[dict(read=p.read, write=p.write) for p in fitted]), model_path)
    receipt = dict(method='parauto', completed=args.step, positions=args.step*32768,
                   training_data_sha256=manifest['train']['sha256'], target_sha256=TARGET_SHA,
                   provenance=provenance, model=upload(model_path, f'milestones/{args.step}/parauto/model.pth'))
    save_json(receipt, destination/'r2_ready.json')
    upload(destination/'r2_ready.json', f'milestones/{args.step}/parauto/r2_ready.json')
    run.summary.update(dict(complete=True, recovered_step=args.step, verification=verifications))
    run.finish()


if __name__ == '__main__': main()
