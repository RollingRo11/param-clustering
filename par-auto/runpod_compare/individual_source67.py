"""Exact component-at-source-position ablations; frozen 67M on Explorer.

Full causal output effects are retained, in float32 after centered float64 KL.
No gradients or output-to-source transfers. See individual_source67.md.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/individual67_20260915'))
from individual67 import (audit, np, torch, MODULES, TARGET_SHA, target, tokens,
                          save_json, sha, credentials, upload, SPECS, BUDGETS,
                          capture, kl_positions)

EXPERIMENT = 'individual_source67_20260915'
FIRST, SEQUENCES, LENGTH = 440, 1, 512
# In the last block these modules cannot influence another output position.
POINTWISE_LAST = {18, 21, 22, 23}


@contextmanager
def subtract_at_source(module, piece, indices, source, scale=1.):
    def hook(_m, args, output):
        a = (args[0][:, source] * piece.read[:, indices].T).sum(-1, keepdim=True)
        result = output.clone()
        result[:, source] -= scale * a * piece.write[:, indices].T
        return result
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def ablated_batch(model, blocks, piece, module_index, indices, source, scale=1., full=False):
    layer = module_index//6
    x = blocks[layer].expand(len(indices), -1, -1)
    with subtract_at_source(model.get_submodule(MODULES[module_index]), piece, indices, source, scale):
        for block in model.h[layer:]:
            x = block(x)
        stop = source+1 if module_index in POINTWISE_LAST else LENGTH
        if not full:
            x = x[:, source:stop]
        return model.lm_head(model.ln_f(x))


@torch.no_grad()
def verify(model, ids, pieces, batch):
    checks = {}
    # Same batch shape for cached and reference prefixes avoids comparing two
    # different cuBLAS accumulation paths. Still explicitly check the KL floor.
    base2, blocks2 = capture(model, ids.expand(2, -1))
    blocks2 = {i: x[:1].clone() for i, x in blocks2.items()}
    for i in [0, 5, 18, 19, 20, 21, 22, 23]:
        p = pieces[i]
        indices = torch.tensor([0, min(17, p.count-1)], device='cuda')
        for source in [0, 255, 511]:
            pred = ablated_batch(model, blocks2, p, i, indices, source, full=True)
            with subtract_at_source(model.get_submodule(MODULES[i]), p, indices, source):
                ref = model(ids.expand(2, -1))[0]
            max_error = float((pred-ref).abs().max())
            floor = float(kl_positions(pred, ref).max())
            assert max_error < 1e-4 and floor < 1e-8, (i, source, max_error, floor)
            effects = kl_positions(pred, base2)
            unaffected = effects[:, :source]
            if i in POINTWISE_LAST:
                unaffected = torch.cat([unaffected, effects[:, source+1:]], -1)
            causality_floor = float(unaffected.max()) if unaffected.numel() else 0.
            assert causality_floor < 1e-8, (i, source, causality_floor)
            checks[f'{i}_{source}'] = dict(logit_max_error=max_error, equivalent_forward_kl=floor,
                                          unaffected_output_kl=causality_floor)
    del base2, blocks2, pred, ref
    base, blocks = capture(model, ids.expand(batch, -1))
    base = base[:1].clone()
    blocks = {i: x[:1].clone() for i, x in blocks.items()}
    indices = torch.arange(batch, device='cuda')
    untouched = ablated_batch(model, blocks, pieces[5], 5, indices, 255, scale=0.)
    zero_floor = float(kl_positions(untouched, base[:, 255:]).max())
    assert zero_floor < 1e-8, zero_floor
    checks['batch_zero_ablation_kl'] = zero_floor
    checks['interpretation'] = 'FP32 forwards; effects near the measured numerical floor are unresolved'
    return checks


@torch.no_grad()
def evaluate_curves(model, ids, pieces, baseline, sequence_scores, final_scores, trainer, out):
    counts = [p.count for p in pieces]
    total = sum(counts)
    results = {}
    for label, scores in [('sequence_global', sequence_scores), ('final_output_global', final_scores),
                          ('sequence_per_source_quota', sequence_scores)]:
        scores = torch.from_numpy(scores).cuda()
        per_source = label.endswith('quota')
        order = scores.argsort(dim=-1, descending=True, stable=True) if per_source else scores.flatten().argsort(descending=True, stable=True)
        entries = []
        for kept in sorted(set(BUDGETS+[total]), reverse=True):
            if kept > total:
                continue
            if per_source:
                mask = audit.shared.support(order, kept)
            else:
                mask = torch.zeros(LENGTH*total, device='cuda', dtype=torch.bool)
                mask[order[:LENGTH*kept]] = True
                mask = mask.reshape(LENGTH, total)
            pred = audit.run(model, ids, pieces, mask.float())
            kl = kl_positions(pred, baseline)[0]
            # The uncharged residual is excluded from the main curve. Keep it
            # in a diagnostic to match exactly the original ablation baseline.
            hooks = []
            with audit.intervention(model, pieces, audit.masks_of(mask.float(), pieces)):
                try:
                    for name, p in zip(MODULES, pieces, strict=True):
                        remainder = (model.get_submodule(name).weight-p.write@p.read.T).detach()
                        def add_remainder(_m, args, output, remainder=remainder):
                            return output+args[0]@remainder.T
                        hooks.append(model.get_submodule(name).register_forward_hook(add_remainder))
                    diagnostic = kl_positions(model(ids)[0], baseline)[0]
                finally:
                    for hook in hooks:
                        hook.remove()
            entries.append(dict(kept_mean_per_source=kept, kept_total=int(mask.sum()),
                                kept_min_per_source=int(mask.sum(-1).min()), kept_max_per_source=int(mask.sum(-1).max()),
                                kl_mean=float(kl.mean()), kl_final=float(kl[-1]), kl_by_output=kl.cpu().tolist(),
                                delta_retained_diagnostic_kl_mean=float(diagnostic.mean()),
                                delta_retained_diagnostic_kl_final=float(diagnostic[-1])))
        results[label] = entries
        save_json(results, out/'curves_partial.json')
    if trainer is not None:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cached = trainer.model(ids, cache_type='input')
            ci = trainer.model.calc_causal_importances(cached.cache, sampling='continuous').lower_leaky
            native = torch.cat([ci[n][0].float() for n in MODULES], -1)>0
        kl = kl_positions(audit.run(model, ids, pieces, native.float()), baseline)[0]
        results['native_ci'] = dict(kept_mean_per_source=float(native.sum(-1).float().mean()),
                                    kl_mean=float(kl.mean()), kl_final=float(kl[-1]))
    return results


def archive(out):
    # Receipts are incremental: interrupted upload passes do not repeat 90GB.
    path = out/'r2_receipts.json'
    receipts = json.loads(path.read_text()) if path.exists() else {}
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.name in ['r2_receipts.json', 'complete.json', 'progress.json'] or p.suffix == '.tmp':
            continue
        rel = str(p.relative_to(out))
        if rel not in receipts:
            receipts[rel] = upload(p, f'experiments/{EXPERIMENT}/{out.name}/{rel}')
            save_json(receipts, path)
    save_json(dict(complete=True, verified_uploads=len(receipts), comparison_sha256=sha(out/'comparison.json')), out/'complete.json')
    upload(out/'complete.json', f'experiments/{EXPERIMENT}/{out.name}/complete.json')


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('family', choices=list(SPECS))
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--archive-only', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    credentials()
    out = ROOT/'output'/EXPERIMENT/args.family
    out.mkdir(parents=True, exist_ok=True)
    if args.archive_only:
        archive(out)
        return 0
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    model = target()
    data, manifest = tokens('val')
    audit.shared.SPECS = SPECS
    pieces, trainer, meta = audit.shared.load(model, args.family)
    counts = [p.count for p in pieces]
    total = sum(counts)
    assert all(c % args.batch == 0 for c in counts)
    plan = dict(family=args.family, checkpoint=meta, target_sha256=TARGET_SHA,
                val_sha256=manifest['val']['sha256'], source_sha256=sha(__file__),
                first=FIRST, sequences=SEQUENCES, context_length=LENGTH, intervention_batch=args.batch,
                components=total, total_individual_interventions=total*LENGTH*SEQUENCES,
                parameters=sum(p.numel() for p in model.parameters()),
                intervention='one original component at exactly one source position in the intact target',
                effects='all potentially affected output positions; earlier causal zeros implicit; float32 KL from float64 reduction',
                ranking='fixed individual effects, cumulative removal least to greatest; global sequence and final-output rankings',
                primary_curve='pieces only; no free Delta; global budgets are mean occurrences per source position',
                note='gim1b is original CE-attribution par-auto, 1B positions / 30518 updates; gim5k uses GIM rules')
    if (out/'plan.json').exists():
        assert json.loads((out/'plan.json').read_text()) == plan
    else:
        save_json(plan, out/'plan.json')
    print(json.dumps(plan), flush=True)
    deadline = time.time()+float(os.environ.get('INDIVIDUAL_SECONDS', '6500'))
    completed, records = 0, []
    started = time.perf_counter()
    for sequence in range(FIRST, FIRST+SEQUENCES):
        ids = torch.tensor(np.array(data[sequence:sequence+1, :-1], dtype=np.int64), device='cuda')
        if not (out/'validation.json').exists():
            checks = verify(model, ids, pieces, args.batch)
            save_json(checks, out/'validation.json')
            print('PREFLIGHT_PASSED', json.dumps(checks), flush=True)
        # Cache a prefix computed at the exact intervention batch shape.
        baseline_batch, blocks = capture(model, ids.expand(args.batch, -1))
        baseline_batch = baseline_batch[:1].clone()
        blocks = {i: x[:1].clone() for i, x in blocks.items()}
        baseline = model(ids)[0]
        batch_floor = float(kl_positions(baseline_batch, baseline).max())
        assert batch_floor < 1e-8, batch_floor
        sequence_scores = np.empty((LENGTH, total), dtype=np.float32)
        final_scores = np.empty_like(sequence_scores)
        files = []
        for source in range(LENGTH):
            offset = 0
            for i, (name, p) in enumerate(zip(MODULES, pieces, strict=True)):
                path = out/f'{sequence}_s{source:03d}_m{i:02d}.npy'
                stop = source+1 if i in POINTWISE_LAST else LENGTH
                if path.exists():
                    values = np.load(path)
                    assert values.shape == (p.count, stop-source)
                else:
                    values = np.empty((p.count, stop-source), dtype=np.float32)
                    began = time.perf_counter()
                    for c in range(0, p.count, args.batch):
                        indices = torch.arange(c, c+args.batch, device='cuda')
                        pred = ablated_batch(model, blocks, p, i, indices, source)
                        values[c:c+args.batch] = kl_positions(pred, baseline_batch[:, source:stop]).cpu().numpy()
                        del pred
                        completed += args.batch
                        if c == 0:
                            print(json.dumps(dict(event='first_batch', source=source, module=name,
                                                  batch_seconds=time.perf_counter()-began)), flush=True)
                    with path.with_suffix('.tmp').open('wb') as f:
                        np.save(f, values)
                    os.replace(path.with_suffix('.tmp'), path)
                    elapsed = time.perf_counter()-started
                    progress = dict(family=args.family, sequence=sequence, source=source, module=name,
                                    interventions_this_invocation=completed, elapsed=elapsed,
                                    interventions_per_second=completed/elapsed,
                                    module_seconds=time.perf_counter()-began, job_id=os.environ['SLURM_JOB_ID'])
                    save_json(progress, out/'progress.json')
                    print(json.dumps(progress), flush=True)
                sequence_scores[source, offset:offset+p.count] = values.sum(-1)/LENGTH
                final_scores[source, offset:offset+p.count] = values[:, -1] if stop == LENGTH else 0.
                files.append(dict(path=path.name, output_start=source, output_stop=stop))
                offset += p.count
                if time.time() > deadline:
                    return 75
        np.save(out/f'{sequence}_sequence_scores.npy', sequence_scores)
        np.save(out/f'{sequence}_final_scores.npy', final_scores)
        result = dict(sequence=sequence, batch_numeric_kl_floor=batch_floor, files=files,
                      curves=evaluate_curves(model, ids, pieces, baseline, sequence_scores, final_scores, trainer, out))
        save_json(result, out/f'{sequence}.json')
        records.append(result)
    save_json(dict(plan=plan, records=records), out/'comparison.json')
    print('SOURCE67_SCORING_COMPLETE', args.family, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
