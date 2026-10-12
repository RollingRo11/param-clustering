"""GIM training where components learn to own rank-one pieces (one gradient step, no alternating merge step).

Pieces: the rank-one GIM setup (1024 / 3072 / 3584 per matrix, seed-0 projected factors, exact-sum projection).
Ownership: per matrix, logits [N, N/OWN_RATIO]; softmax per piece gives each component's share of that piece.
Component effect = ownership-weighted sum of its pieces' signed effects; loss per token = sum over components of
score^p * (pieces owned)^OWN_ALPHA, normalized at the start. Gradient flows into pieces and ownership together.
Evaluation assigns each piece to its argmax component and masks whole components (empty ones drop out).
Everything else as the lr=1e-3, p_end=.5 GIM run; 5,000 steps of the 8,700-step horizon.
"""
from dataclasses import asdict
import json
import os
import random
import signal
import time
import numpy as np
import torch
import wandb
from common import ROOT, MODULES, TARGET_SHA, credentials, tokens, batch, target, checkpoint, save_json
from train_pair import core, method
from gim_attribution import capture_gim, RULES

RATIO = int(os.environ.get('OWN_RATIO', '4'))
ALPHA = float(os.environ.get('OWN_ALPHA', '0.2'))
HARD = os.environ.get('OWN_HARD', '0') == '1'
OUT = ROOT/f'output/gim_own{RATIO}_a{ALPHA:g}{"_st" if HARD else ""}_5000'
BINS = [0, 1, 2, 3, 5, 9, 17, 65, 10**9]


def ownership_summary(ownership):
    out = {}
    for name, logits in zip(MODULES, ownership):
        mix = torch.softmax(logits.float(), -1)
        sizes = torch.bincount(mix.argmax(-1), minlength=mix.shape[1]).cpu()
        hist = {f'{BINS[i]}-{BINS[i+1]-1}' if BINS[i+1] < 10**9 else f'{BINS[i]}+': int(((sizes >= BINS[i]) & (sizes < BINS[i+1])).sum())
                for i in range(len(BINS)-1)}
        out[name] = dict(slots=int(mix.shape[1]), used=int((sizes > 0).sum()), size_max=int(sizes.max()),
                         size_histogram=hist, ownership_max_mean=float(mix.max(-1).values.mean()))
    return out


def main():
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists(): raise FileExistsError('Refusing to overwrite an existing ownership run')
    deadline = float(os.environ['GIM_DEADLINE'])
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGUSR1, request_stop)
    data, manifest = tokens(); model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    counts = [dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    initial = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, counts)]
    index = torch.randperm(32768, device='cuda', generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    control_path = ROOT/'output/gim_1h/plan.json'
    control = json.loads(control_path.read_text())
    assert control['sensor'] == 'gim' and control['rules'] == RULES
    assert control['target_sha256'] == TARGET_SHA and control['data_sha256'] == manifest['train']['sha256']
    assert control['component_counts'] == dict(zip(MODULES, counts)) and control['reference_index_seed'] == 123
    cfg = method.TrainConfig(**control['training_config'])
    assert cfg.p_end == .5 and cfg.steps == 8700 and cfg.lr == 1e-3
    cfg.ownership_ratio, cfg.rank_price, cfg.ownership_hard = RATIO, ALPHA, HARD
    base = {**asdict(method.TrainConfig()), **control['training_config']}
    changed = {k for k, v in asdict(cfg).items() if v != base[k]}
    assert changed == {'ownership_ratio', 'rank_price'} | ({'ownership_hard'} if HARD else set()), changed
    first = capture_gim(model, batch(data, 0))
    reference = [method.slice_entry(e, index) for e in first]
    del first
    plan = dict(sensor='gim', rules=RULES, training_config=asdict(cfg), stop_after=5000,
                ownership_ratio=RATIO, rank_price=ALPHA, pieces_per_matrix=dict(zip(MODULES, counts)),
                component_slots_per_matrix=dict(zip(MODULES, [c//RATIO for c in counts])),
                control_plan=str(control_path), control_training_config=control['training_config'],
                changed_settings={'ownership_ratio': [0, RATIO], 'rank_price': [0., ALPHA], 'ownership_hard': [False, HARD]},
                note=('Straight-through ownership: argmax in the forward pass (as at evaluation), softmax gradient backward.' if HARD else
                      'Learned soft ownership of rank-one pieces by components, trained jointly; hard argmax ownership at evaluation.'),
                deadline_unix=deadline, checkpoint_interval=1000, target_sha256=TARGET_SHA,
                data_sha256=manifest['train']['sha256'], gpu=torch.cuda.get_device_name(), reference_index_seed=123,
                job_id=os.environ.get('SLURM_JOB_ID'))
    save_json(plan, OUT/'plan.json'); print(json.dumps(dict(plan=plan)), flush=True)
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu', group='ownership-20260913',
                     name=f'parauto-gim-own{RATIO}-a{ALPHA:g}{"-st" if HARD else ""}-5000', job_type='training', dir=str(OUT), config=plan)
    save_json(dict(id=run.id, url=run.url), OUT/'wandb.json')
    run.define_metric('update'); run.define_metric('train/*', step_metric='update')
    completed = 0; started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    def progress(value):
        nonlocal completed
        completed = value
        if value % 100 == 0:
            torch.cuda.synchronize()
            row = dict(update=value, positions=value*32768, seconds=time.perf_counter()-started,
                       seconds_per_step=(time.perf_counter()-started)/value,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), remaining_seconds=max(0., deadline-time.time()))
            save_json(row, OUT/'progress.json')
            run.log({'update': value, **{'train/'+k: v for k, v in row.items() if k != 'update'}})
            print(json.dumps(row), flush=True)
        return stop or time.time() >= deadline
    def save(state):
        step = state['next_step']; before = time.perf_counter()
        out = OUT/'checkpoints'/f'{step:06d}'
        identity = dict(method='parauto', sensor='gim', completed=step, training_positions=step*32768,
                        data_sha256=manifest['train']['sha256'], target_sha256=TARGET_SHA)
        checkpoint(dict(**identity, state=state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), out/'checkpoint.pt')
        factors = [dict(**piece, ownership_logits=logits) for piece, logits in zip(state['pieces'], state['ownership'])]
        checkpoint(dict(**identity, factors=factors, provenance=dict(kind=f'GIM training, learned ownership (ratio {RATIO}, rank price {ALPHA:g}, straight-through {HARD})',
                        rules=RULES, training_config=asdict(cfg))), out/'model.pth')
        summary = ownership_summary(state['ownership'])
        save_json(summary, out/'ownership_summary.json')
        record = dict(step=step, model=str(out/'model.pth'), seconds=time.perf_counter()-before,
                      components_used=sum(v['used'] for v in summary.values()), pieces=sum(counts),
                      ownership_max_mean=float(np.mean([v['ownership_max_mean'] for v in summary.values()])))
        save_json(record, OUT/'latest_checkpoint.json')
        print(json.dumps(dict(checkpoint=record)), flush=True)
        run.log({'update': step, 'train/checkpoint_seconds': record['seconds'], 'train/components_used_all': record['components_used'],
                 'train/ownership_max_mean_all': record['ownership_max_mean']})
    def log(row):
        i = row['matrix_index']
        values = {f'components_used/m{i}': row.get('components_used', np.nan), f'component_size_max/m{i}': row.get('component_size_max', np.nan),
                  f'ownership_max_mean/m{i}': row.get('ownership_max_mean', np.nan), f'loss/m{i}': row['loss']}
        if i == 0:
            values.update(total_loss=row['total_loss'], p=row['p'], lr=core.cosine_lr(row['step'], cfg))
            if not np.isfinite([row['total_loss'], row['p']]).all(): raise RuntimeError('Nonfinite objective')
        run.log({'update': row['step']+1, **{'train/'+k: v for k, v in values.items()}})
    method.fit(reference, weights, initial, cfg, batch_provider=lambda step: capture_gim(model, batch(data, step)),
               step_callback=progress, checkpoint=save, checkpoint_interval=1000, stop_after=5000, log=log)
    torch.cuda.synchronize()
    result = dict(completed=completed, positions=completed*32768, seconds=time.perf_counter()-started,
                  stop_reason='signal' if stop else 'time_budget' if time.time() >= deadline else 'target_steps',
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), training_config=asdict(cfg),
                  data_sha256=manifest['train']['sha256'], target_sha256=TARGET_SHA)
    save_json(result, OUT/'result.json'); run.summary.update(result); run.finish()
    print(json.dumps(dict(finished=result)), flush=True)
    if completed != 5000: raise RuntimeError('Incomplete run; do not publish a matched-step evaluation')


if __name__ == '__main__': main()
