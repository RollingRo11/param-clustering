"""GIM training with rank-k pieces: each piece is one matrix U_c R_c^T of rank <= VR_RANK (default 8).

Same columns per matrix as the lr=1e-3, p_end=.5 GIM run (1024 / 3072 / 3584), so pieces per matrix = columns / rank,
the same seed-0 projected factors and the same exact-sum projection. Changed settings: piece_rank (whole-piece
first-order score, whole-piece sparsity) and rank_coeff (env VR_RANK_COEFF, default 1: activity-weighted Lp of column
sizes, which lowers each piece's rank). 5,000 steps of the 8,700-step horizon, like the other 5k GIM runs.
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

RANK = int(os.environ.get('VR_RANK', '8'))
COEFF = float(os.environ.get('VR_RANK_COEFF', '1'))
OUT = ROOT/f'output/gim_rank{RANK}_c{COEFF:g}_5000'


def rank_summary(pieces, rank):
    out = {}
    for name, p in zip(MODULES, pieces):
        sv = core.piece_singular_values(core.FreePieces(p['read'].cuda(), p['write'].cuda()), rank)
        eff = (sv > 1e-2*sv[:, :1].clamp_min(1e-30)).sum(-1)
        out[name] = dict(pieces=int(sv.shape[0]), effective_rank_histogram=torch.bincount(eff, minlength=rank+1).tolist(),
                         participation_rank_mean=float((sv.sum(-1).square()/sv.square().sum(-1).clamp_min(1e-30)).mean()),
                         top_singular_share_mean=float((sv[:, 0]/sv.sum(-1).clamp_min(1e-30)).mean()))
    return out


def main():
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'plan.json').exists(): raise FileExistsError('Refusing to overwrite an existing rank-k run')
    deadline = float(os.environ['GIM_DEADLINE'])
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGUSR1, request_stop)
    data, manifest = tokens(); model = target()
    weights = [model.get_submodule(n).weight for n in MODULES]
    columns = [dict(q_proj=1024, k_proj=1024, v_proj=1024, o_proj=1024, c_fc=3072, down_proj=3584)[n.split('.')[-1]] for n in MODULES]
    assert all(c % RANK == 0 for c in columns)
    initial = [core.initialize_pieces(w, c, 0) for w, c in zip(weights, columns)]
    index = torch.randperm(32768, device='cuda', generator=torch.Generator(device='cuda').manual_seed(123))[:256]
    control_path = ROOT/'output/gim_1h/plan.json'
    control = json.loads(control_path.read_text())
    assert control['sensor'] == 'gim' and control['rules'] == RULES
    assert control['target_sha256'] == TARGET_SHA and control['data_sha256'] == manifest['train']['sha256']
    assert control['component_counts'] == dict(zip(MODULES, columns)) and control['reference_index_seed'] == 123
    cfg = method.TrainConfig(**control['training_config'])
    assert cfg.p_end == .5 and cfg.steps == 8700 and cfg.lr == 1e-3
    cfg.piece_rank, cfg.rank_coeff = RANK, COEFF
    base = {**asdict(method.TrainConfig()), **control['training_config']}
    changed = {k for k, v in asdict(cfg).items() if v != base[k]}
    assert changed <= {'piece_rank', 'rank_coeff'} and cfg.piece_rank > 1, changed
    first = capture_gim(model, batch(data, 0))
    reference = [method.slice_entry(e, index) for e in first]
    del first
    plan = dict(sensor='gim', rules=RULES, training_config=asdict(cfg), stop_after=5000,
                piece_rank=RANK, rank_coeff=COEFF, columns_per_matrix=dict(zip(MODULES, columns)),
                pieces_per_matrix=dict(zip(MODULES, [c//RANK for c in columns])),
                control_plan=str(control_path), control_training_config=control['training_config'],
                changed_settings={'piece_rank': [1, RANK], 'rank_coeff': [0., COEFF]},
                note='Each piece is one rank<=k matrix: scored, penalized and masked as a whole. Columns, init and projection as control.',
                deadline_unix=deadline, checkpoint_interval=1000, target_sha256=TARGET_SHA,
                data_sha256=manifest['train']['sha256'], gpu=torch.cuda.get_device_name(), reference_index_seed=123,
                job_id=os.environ.get('SLURM_JOB_ID'))
    save_json(plan, OUT/'plan.json'); print(json.dumps(dict(plan=plan)), flush=True)
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu', group='variable-rank-20260912',
                     name=f'parauto-gim-rank{RANK}-c{COEFF:g}-5000', job_type='training', dir=str(OUT), config=plan)
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
        checkpoint(dict(**identity, factors=state['pieces'], provenance=dict(kind=f'GIM training, rank-{RANK} pieces',
                        rules=RULES, training_config=asdict(cfg))), out/'model.pth')
        summary = rank_summary(state['pieces'], RANK)
        save_json(summary, out/'rank_summary.json')
        record = dict(step=step, model=str(out/'model.pth'), seconds=time.perf_counter()-before,
                      participation_rank_mean=float(np.mean([v['participation_rank_mean'] for v in summary.values()])))
        save_json(record, OUT/'latest_checkpoint.json')
        print(json.dumps(dict(checkpoint=record)), flush=True)
        run.log({'update': step, 'train/checkpoint_seconds': record['seconds'], 'train/participation_rank_all': record['participation_rank_mean']})
    def log(row):
        values = {f'effective_rank/m{row["matrix_index"]}': row.get('effective_rank_mean', np.nan),
                  f'rank_normalized/m{row["matrix_index"]}': row.get('rank_normalized', np.nan)}
        if row['matrix_index'] == 0:
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
