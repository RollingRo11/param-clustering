"""Three separate 67M Pile training runs, with prerequisite validation.

Entry points are SLURM-only. Sources and the upstream model implementation
are hash checked by the launcher. Interrupted runs resume at an optimizer
boundary; explicit step/microbatch seeds regenerate every stochastic choice.
"""
import argparse
from dataclasses import asdict, replace
import gc
import io
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import ROOT, TARGET_SHA, target, tokens, batch, checkpoint, save_json, sha, credentials, upload
import engine as e

EXPERIMENT = 'conditional67_20260916_v3'
HERE = Path(__file__).resolve().parent


def identity(cfg, variant, manifest):
    return dict(experiment=EXPERIMENT, variant=variant, config=asdict(cfg),
                target_sha256=TARGET_SHA, training_data_sha256=manifest['train']['sha256'],
                source_manifest_sha256=sha(HERE/'SHA256SUMS'),
                components=[1024, 1024, 1024, 1024, 3072, 3584]*4,
                initialization='Fresh seed-0 random factors, exact sum, no warm start',
                factor_objective='Actual sparse-output KL; unconditional projected Adam after accumulated 64-context gradient',
                selection_objective='Per-sequence KL + 0.001 * mean retained components per prediction position',
                support='Binary; common sampled-label K1 initialization; variant-specific proposals; finite objective acceptance',
                schedule='5000-step cosine LR; budget anneals 8192->256 over 4000 updates, then cycles 128/192/256/512',
                training_positions=cfg.steps*cfg.contexts*512, mask_free_training=False,
                backbone='Frozen embeddings, norms, biases, residual architecture and output head; all 24 matrix sums exact',
                cost_note='Whole selected model, dense factor GEMMs; no claim of runtime proportional to retained count')


def make_pieces(model):
    return [e.core.initialize_pieces(model.get_submodule(n).weight, c, seed=0)
            for n, c in zip(e.names(model), [1024, 1024, 1024, 1024, 3072, 3584]*4, strict=True)]


@torch.no_grad()
def weight_errors(model, pieces):
    errors = [e.core.reconstruction_error(model.get_submodule(n).weight, p) for n, p in zip(e.names(model), pieces, strict=True)]
    assert max(errors) < 1e-5, errors
    return errors


def state(pieces, optimizer, completed):
    return dict(factors=[dict(read=p.read.detach(), write=p.write.detach()) for p in pieces],
                optimizer=optimizer.state_dict(), completed=completed,
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def from_state(saved, cfg):
    pieces = [e.core.FreePieces(p['read'].cuda().requires_grad_(), p['write'].cuda().requires_grad_()) for p in saved['factors']]
    optimizer = e.create_optimizer(pieces, cfg)
    optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return pieces, optimizer


def check_derivatives(model, pieces, rows, cfg):
    exact = replace(cfg, precision='fp32')
    ids = rows[:1, :-1]
    with torch.no_grad():
        baseline = model(ids)[0].float()
        all_on = torch.ones(*ids.shape, sum(p.count for p in pieces), device=ids.device)
        rebuilt = e.run(model, ids, pieces, all_on, exact)
        all_kl = float(e.kl_per_sequence(rebuilt, baseline.log_softmax(-1)).mean())
        relative = float((rebuilt-baseline).norm()/baseline.norm())
        assert relative < 2e-4 and abs(all_kl) < 1e-5, (relative, all_kl)
        del baseline, rebuilt, all_on
    logp, keep = e.dense_reference(model, ids, pieces, exact, 780123, 512)
    gate = keep.float().requires_grad_()
    loss = e.kl_per_sequence(e.run(model, ids, pieces, gate, exact), logp).sum()
    grad, = torch.autograd.grad(loss, gate)
    # One strongly measurable perturbation for each matrix type. Both off
    # insertions and on deletions are additionally covered by CPU tests.
    records = []
    offsets = np.cumsum([0]+[p.count for p in pieces]).tolist()
    for kind in range(6):
        direction = torch.zeros_like(gate)
        block = grad[..., offsets[kind]:offsets[kind+1]]
        idx = block.abs().flatten().topk(64).indices
        local = torch.zeros_like(block).flatten()
        local[idx] = block.flatten()[idx].sign()
        direction[..., offsets[kind]:offsets[kind+1]] = local.reshape_as(block)
        predicted = float((grad*direction).sum())
        measurements = []
        for eps in (.03, .01, .003):
            with torch.no_grad():
                plus = e.run(model, ids, pieces, gate.detach()+eps*direction, exact)
                minus = e.run(model, ids, pieces, gate.detach()-eps*direction, exact)
                observed = float(e.stable_kl_difference(plus, minus, logp).sum()/(2*eps))
            discrepancy = abs(predicted-observed)/max(abs(predicted), abs(observed), 1e-5)
            measurements.append(dict(epsilon=eps, observed=observed, relative_error=discrepancy))
        # Require BOTH smaller steps to pass the original tolerance. This
        # improves numerical conditioning rather than relaxing the check.
        assert all(m['relative_error'] < .04 for m in measurements[1:]), (kind, predicted, measurements)
        records.append(dict(kind=e.KINDS[kind], predicted=predicted, measurements=measurements))
    return dict(all_on_relative_logits=relative, all_on_kl=all_kl, finite_differences=records)


def prepare(model, training, initial, cfg, variant, out, plan):
    rows = batch(training, 0)
    print('VALIDATION derivatives '+variant, flush=True)
    checks = check_derivatives(model, initial, rows, cfg)
    # Actual persisted state is CPU serialized, including Adam step tensors.
    small = replace(cfg, contexts=4, microbatch=2, start_budget=512)
    pieces = e.clone_pieces(initial)
    optimizer = e.create_optimizer(pieces, small)
    e.train_step(model, rows[:4], pieces, optimizer, small, variant, 0)
    path = out/'validation_resume.pt'
    checkpoint(state(pieces, optimizer, 1), path)
    e.train_step(model, rows[:4], pieces, optimizer, small, variant, 1)
    resumed, ropt = from_state(torch.load(path, map_location='cpu', weights_only=False), small)
    e.train_step(model, rows[:4], resumed, ropt, small, variant, 1)
    for a, b in zip(pieces, resumed, strict=True):
        torch.testing.assert_close(a.read, b.read, rtol=0, atol=0)
        torch.testing.assert_close(a.write, b.write, rtol=0, atol=0)
    checks['resume_bitwise'] = True
    checks['max_matrix_error'] = max(weight_errors(model, pieces))
    del pieces, optimizer, resumed, ropt
    path.unlink()
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    # Full production batch at a moderately sparse and a final-budget step.
    # All pilot updates are discarded before fresh training starts.
    pieces = e.clone_pieces(initial)
    optimizer = e.create_optimizer(pieces, cfg)
    timings, metrics = [], []
    for step in (0, 3999):
        print('VALIDATION full_batch '+str(step), flush=True)
        torch.cuda.synchronize(); start = time.perf_counter()
        result = e.train_step(model, rows, pieces, optimizer, cfg, variant, step)
        torch.cuda.synchronize(); timings.append(time.perf_counter()-start)
        metrics.append(result)
        weight_errors(model, pieces)
    assert any(not torch.equal(a.read, b.read) or not torch.equal(a.write, b.write) for a, b in zip(initial, pieces, strict=True))
    checks.update(passed=True, full_batch_seconds=timings, full_batch_metrics=metrics,
                  estimated_5000_hours=5000*statistics.mean(timings)/3600,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  source_manifest_sha256=plan['source_manifest_sha256'])
    save_json(checks, out/'prepare.json')
    print('PREPARE_COMPLETE '+json.dumps(checks), flush=True)


def train(model, training, cfg, variant, out, plan):
    prepared = json.loads((out/'prepare.json').read_text())
    assert prepared['passed'] and prepared['source_manifest_sha256'] == plan['source_manifest_sha256']
    path = out/'resume.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['plan'] == json.loads(json.dumps(plan)), 'Resume provenance/config mismatch'
        pieces, optimizer = from_state(saved['state'], cfg)
        completed = saved['state']['completed']
        del saved
    else:
        initial = make_pieces(model)
        pieces = e.clone_pieces(initial); del initial
        optimizer = e.create_optimizer(pieces, cfg)
        completed = 0
    if completed == cfg.steps:
        return
    import wandb
    previous = json.loads((out/'wandb.json').read_text()) if (out/'wandb.json').exists() else None
    run = wandb.init(project='param-autoenc', entity='rohan-kathuria-neu', group=EXPERIMENT,
                     name=variant+'-5000', dir=str(out), config=plan,
                     id=previous['id'] if previous else None, resume='must' if previous else None)
    save_json(dict(id=run.id, url=run.url), out/'wandb.json')
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGUSR1, stopping); signal.signal(signal.SIGTERM, stopping)
    deadline = float(os.environ.get('CONDITIONAL_DEADLINE', 'inf'))
    start_step = completed; began = time.perf_counter()
    def save():
        errors = weight_errors(model, pieces)
        checkpoint(dict(plan=plan, state=state(pieces, optimizer, completed)), path)
        modelpath = out/'checkpoints'/f'{completed:06d}'/'model.pth'
        checkpoint(dict(method='parauto', sensor=variant, completed=completed,
                        target_sha256=TARGET_SHA, data_sha256=plan['training_data_sha256'],
                        factors=[dict(read=p.read, write=p.write) for p in pieces], provenance=plan), modelpath)
        save_json(dict(completed=completed, checkpoint=str(path), model=str(modelpath),
                       max_matrix_error=max(errors)), out/'latest.json')
    try:
        for step in range(completed, cfg.steps):
            rows = batch(training, step)
            assert rows.shape == (cfg.contexts, 513)
            metrics = e.train_step(model, rows, pieces, optimizer, cfg, variant, step)
            completed = step+1
            metrics.update(update=completed, positions=completed*cfg.contexts*512,
                           seconds_per_step=(time.perf_counter()-began)/(completed-start_step))
            if completed % 10 == 0 or completed == start_step+1:
                run.log(metrics, step=completed)
                save_json(metrics, out/'progress.json')
                with (out/'history.jsonl').open('a') as f:
                    f.write(json.dumps(metrics, allow_nan=False)+'\n')
                print(json.dumps(metrics), flush=True)
            if completed % 250 == 0 or stop or time.time() > deadline:
                save()
            if stop or time.time() > deadline:
                break
        if completed == cfg.steps:
            save_json(dict(complete=True, completed=completed, source_manifest_sha256=plan['source_manifest_sha256']), out/'training_complete.json')
    finally:
        run.finish()
    print('TRAINING_STOP '+str(completed), flush=True)
    if completed < cfg.steps:
        raise SystemExit(75)


def evaluate(model, cfg, variant, out, *, smoke=False):
    latest = json.loads((out/'latest.json').read_text())
    saved = torch.load(latest['model'], map_location='cpu', weights_only=False)
    assert saved['target_sha256'] == TARGET_SHA
    pieces = [e.core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    data, _ = tokens('val')
    first_row, sequences = (1000, 4) if smoke else (1024, 128)
    assert len(data) >= first_row+sequences
    budgets = (192, 512) if smoke else (64, 128, 192, 256, 512, 1024, 4096)
    records = []
    for first in range(first_row, first_row+sequences, cfg.microbatch):
        rows = torch.tensor(np.array(data[first:first+cfg.microbatch], dtype=np.int64), device='cuda')
        ids = rows[:, :-1]
        # Common precise ranking is a distinct question from each arm's
        # actual cheap K1-seeded native selection. Save both independently.
        logp, _, fisher = e.dense_reference(model, ids, pieces, cfg, 71000000+first, max(budgets),
                                           probe_count=4 if smoke else 64, return_scores=True)
        _, _, cheap = e.dense_reference(model, ids, pieces, cfg, 71000000+first, max(budgets), return_scores=True)
        for budget in budgets:
            ranked = e.top_support(fisher, budget)
            initial = e.top_support(cheap, budget)
            native, info = e.select(model, ids, pieces, initial, logp, cfg, variant, 72000000+first)
            for label, support in [('fisher', ranked), ('native', native)]:
                with torch.no_grad():
                    logits = e.run(model, ids, pieces, support.float(), cfg)
                    kl = e.kl_per_sequence(logits, logp)
                    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), rows[:, 1:].flatten(), reduction='none').reshape_as(ids).mean(-1)
                for j in range(len(rows)):
                    records.append(dict(row=first+j, ranking=label, budget=budget, kl=float(kl[j]), ce=float(ce[j]),
                                        kept_per_token=float(support[j].sum(-1).float().mean())))
        print('EVAL_ROWS '+str(first), flush=True)
    result = dict(variant=variant, completed=latest['completed'], model_sha256=sha(latest['model']),
                  first_row=first_row, sequences=sequences, fisher_probes=4 if smoke else 64,
                  native_initialization_probes=1, records=records)
    save_json(result, out/('smoke_eval.json' if smoke else 'evaluation.json'))


def archive(out):
    latest = json.loads((out/'latest.json').read_text())
    files = [Path(latest['model']), out/'resume.pt']
    files += [out/n for n in ('plan.json', 'prepare.json', 'progress.json', 'history.jsonl',
                              'wandb.json', 'training_complete.json', 'evaluation.json') if (out/n).exists()]
    records = []
    for path in files:
        relative = path.relative_to(out)
        records.append(upload(path, f'experiments/{EXPERIMENT}/{out.name}/{relative}'))
    save_json(dict(verified_uploads=records), out/('archive_evaluated.json' if (out/'evaluation.json').exists() else 'archive_initial.json'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('prepare', 'train', 'eval', 'smoke-eval', 'archive'))
    ap.add_argument('variant', choices=e.VARIANTS)
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), '67M numerical work must run in SLURM'
    credentials(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    cfg = e.Config()
    out = ROOT/'output'/EXPERIMENT/args.variant
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'archive':
        archive(out); return
    model = target()
    assert sum(p.numel() for p in model.parameters()) == 66931200
    if args.mode in ('eval', 'smoke-eval'):
        evaluate(model, cfg, args.variant, out, smoke=args.mode == 'smoke-eval'); return
    training, manifest = tokens()
    assert len(training) >= cfg.steps*cfg.contexts
    plan = json.loads(json.dumps(identity(cfg, args.variant, manifest)))
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == json.loads(json.dumps(plan)), 'Immutable experiment configuration changed'
    else:
        save_json(plan, out/'plan.json')
    if args.mode == 'prepare':
        prepare(model, training, make_pieces(model), cfg, args.variant, out, plan)
    else:
        train(model, training, cfg, args.variant, out, plan)


if __name__ == '__main__':
    main()
