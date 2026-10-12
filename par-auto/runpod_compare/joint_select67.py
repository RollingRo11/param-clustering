"""Frozen 67M, binary support refinement using full-sequence signed Fisher effects."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['PAIR_ROOT'])
sys.path.insert(0, str(ROOT/'code/gim_count67_20260914/compare'))
import numpy as np
import torch
from common import MODULES, TARGET_SHA, target, tokens, sha, save_json, credentials, upload
from train_pair import VPD, core
from fisher_attribution import capture_fisher

SPECS = {
    'gim': ('parauto', 5000, 'output/gim_1h/checkpoints/005000/model.pth'),
    'count': ('parauto', 5000, 'output/gim_count67_20260914/checkpoints/005000/model.pth'),
    'vpd': ('vpd', 400000, 'checkpoints/published_vs_finished/vpd.pth'),
}
OUT = ROOT/'output/joint_select67_20260915'


@contextmanager
def replace(model, pieces, masks, layers=(0, 1, 2, 3)):
    hooks = []
    try:
        for name, p, mask in zip(MODULES, pieces, masks, strict=True):
            if int(name.split('.')[1]) not in layers:
                continue
            def hook(layer, inputs, output, p=p, mask=mask):
                value = ((inputs[0] @ p.read)*mask) @ p.write.T
                return value if layer.bias is None else value+layer.bias
            hooks.append(model.get_submodule(name).register_forward_hook(hook))
        yield
    finally:
        for hook in hooks:
            hook.remove()


def forward(model, ids, pieces, masks, **kwargs):
    with replace(model, pieces, masks, **kwargs), torch.autocast('cuda', dtype=torch.bfloat16):
        return model(ids)[0]


def kl(logits, baseline):
    lp = baseline.float().log_softmax(-1)
    return float((lp.exp()*(lp-logits.float().log_softmax(-1))).sum(-1).mean())


def support(order, k):
    return torch.zeros_like(order, dtype=torch.bool).scatter_(-1, order[:, :k], True)


def residual(z, reference, keep):
    return reference - torch.einsum('tck,tc->k', z, keep.float())


@torch.no_grad()
def refine(z, reference, initial, seed, sweeps=4, candidates=8):
    """Exact accepted swap updates, bounded candidate search, fixed count per token.

    Candidate shortlist takes the best individual add/drop moves, then tests all
    cross pairs. This is a local search, not an oracle or global optimum.
    """
    keep = initial.clone()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    norm = z.square().sum(-1)
    r = residual(z, reference, keep)
    history = [float(r.square().sum())]
    accepted = []
    for sweep in range(sweeps):
        moves = 0
        for t in torch.randperm(z.shape[0], generator=generator).tolist():
            dot = z[t] @ r
            drop_score = (2*dot+norm[t]).masked_fill(~keep[t], float('inf'))
            add_score = (-2*dot+norm[t]).masked_fill(keep[t], float('inf'))
            d = drop_score.topk(candidates, largest=False).indices
            a = add_score.topk(candidates, largest=False).indices
            delta = z[t, d, None, :] - z[t, None, a, :]
            change = 2*torch.einsum('dak,k->da', delta, r)+delta.square().sum(-1)
            best = change.flatten().argmin()
            if float(change.flatten()[best]) < -1e-7:
                di, ai = best//candidates, best % candidates
                r += delta[di, ai]
                keep[t, d[di]] = False
                keep[t, a[ai]] = True
                moves += 1
        exact = residual(z, reference, keep)
        drift = float((r-exact).norm()/exact.norm().clamp_min(1e-8))
        r = exact
        current = float(r.square().sum())
        assert current <= history[-1]+max(1e-3, history[-1]*1e-4), (history[-1], current, drift)
        history.append(current)
        accepted.append(dict(swaps=moves, residual_drift=drift))
    assert torch.equal(keep.sum(-1), initial.sum(-1))
    return keep, dict(objective_history=history, sweeps=accepted,
                      changed_per_token=float((keep & ~initial).sum(-1).float().mean()))


@torch.no_grad()
def validate_search():
    # Non-model algebra check: includes an unrepresented Delta in reference.
    gen = torch.Generator(device='cuda').manual_seed(720)
    z = torch.randn(5, 24, 9, device='cuda', generator=gen)
    reference = z.sum((0, 1))+torch.randn(9, device='cuda', generator=gen)
    initial = support(z.square().sum(-1).argsort(-1, descending=True), 8)
    found, record = refine(z, reference, initial, 80, sweeps=2, candidates=4)
    torch.testing.assert_close(residual(z, reference, found), reference-z[found].sum(0), rtol=1e-5, atol=1e-5)
    assert record['objective_history'][-1] <= record['objective_history'][0]
    return record


def load(model, name):
    kind, step, rel = SPECS[name]
    path = ROOT/rel
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['method'] == kind and saved['completed'] == step and saved['target_sha256'] == TARGET_SHA
    trainer = None
    if kind == 'vpd':
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True)
        handle_deprecated_state_dict_keys_(saved['factors'])
        for n, v in model.state_dict().items():
            torch.testing.assert_close(saved['factors']['target_model.'+n], v.cpu(), rtol=0, atol=0)
        trainer.model.load_state_dict(saved['factors'])
        trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(),
                                 trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda().float(), p['write'].cuda().float()) for p in saved['factors']]
    errors = [core.reconstruction_error(model.get_submodule(n).weight, p) for n, p in zip(MODULES, pieces)]
    if kind == 'parauto':
        assert max(errors) < 1e-5
    return pieces, trainer, dict(checkpoint=str(path), checkpoint_sha256=sha(path), step=step,
                                 matrix_sum_errors=errors)


@torch.no_grad()
def make_effects(model, cache, pieces):
    t, k = cache[0]['events']['cot'].shape[:2]
    z = torch.empty(t, sum(p.count for p in pieces), k, device='cuda', dtype=torch.float32)
    reference = torch.zeros(k, device='cuda')
    start = 0
    for name, e, p in zip(MODULES, cache, pieces, strict=True):
        h, g = e['h'].float(), e['events']['cot'].float()
        read = h @ p.read
        # FP32 scoring prevents BF16 cancellation noise from defining the search.
        effect = torch.einsum('tkd,dc->tck', g, p.write)
        z[:, start:start+p.count] = read[..., None]*effect
        reference += torch.einsum('tkd,td->k', g, h @ model.get_submodule(name).weight.T)
        start += p.count
    return z, reference


@torch.no_grad()
def assess(model, ids, baseline, cache, pieces, keep):
    counts = [p.count for p in pieces]
    masks = [v[None].float() for v in keep.split(counts, -1)]
    total = torch.zeros(cache[0]['events']['cot'].shape[1], device='cuda')
    relative = []
    per_matrix = []
    for name, e, p, mask in zip(MODULES, cache, pieces, masks, strict=True):
        h, g = e['h'].float(), e['events']['cot'].float()
        clean = h @ model.get_submodule(name).weight.T
        delta = ((h @ p.read)*mask[0]) @ p.write.T-clean
        total += torch.einsum('tkd,td->k', g, delta)
        relative.append(float(delta.norm()/clean.norm().clamp_min(1e-12)))
        per_matrix.append(float(mask.sum()/ids.numel()))
    return dict(kl=kl(forward(model, ids, pieces, masks), baseline),
                independent_quadratic=float(.5*total.square().mean()/ids.numel()),
                restore_last_kl=kl(forward(model, ids, pieces, masks, layers=(0, 1, 2)), baseline),
                clean_input_relative_error=relative, per_matrix_count=per_matrix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first', type=int, default=400)
    ap.add_argument('--sequences', type=int, default=16)
    ap.add_argument('--fit-k', type=int, default=128)
    ap.add_argument('--check-k', type=int, default=128)
    ap.add_argument('--sweeps', type=int, default=4)
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Explorer allocation required'
    credentials(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    OUT.mkdir(parents=True, exist_ok=True)
    validation = validate_search()
    model = target(); data, manifest = tokens('val')
    plan = dict(**vars(args), budgets=[128, 256, 512], starts=['magnitude', 'jittered_magnitude'],
                shortlist=8, source_sha256=sha(__file__), target_sha256=TARGET_SHA,
                data_sha256=manifest['val']['sha256'], parameters=sum(p.numel() for p in model.parameters()),
                fit_seed_offset=3_100_000, check_seed_offset=4_100_000,
                note='No factor training; fixed binary count per token; chosen by fit sketch only; full sequence residual includes Delta.')
    if (OUT/'plan.json').exists():
        assert json.loads((OUT/'plan.json').read_text()) == plan
    else:
        save_json(plan, OUT/'plan.json')
    save_json(validation, OUT/'search_validation.json')
    deadline = float(os.environ.get('JOINT_DEADLINE', 'inf'))
    began = time.perf_counter()
    for first in range(args.first, args.first+args.sequences):
        if all((OUT/f'{first}_{n}.json').exists() for n in SPECS):
            continue
        if time.time() > deadline:
            return 75
        rows = torch.tensor(np.array(data[first:first+1], dtype=np.int64), device='cuda')
        ids = rows[:, :-1]
        cache = capture_fisher(model, rows, k=args.fit_k, seed=3_100_000+first, microbatch=1)
        check = capture_fisher(model, rows, k=args.check_k, seed=4_100_000+first, microbatch=1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            baseline = model(ids)[0]
        for name in SPECS:
            dest = OUT/f'{first}_{name}.json'
            if dest.exists():
                continue
            if time.time() > deadline:
                return 75
            pieces, trainer, metadata = load(model, name)
            z, reference = make_effects(model, cache, pieces)
            scores = z.square().mean(-1).sqrt()
            order = scores.argsort(dim=-1, descending=True, stable=True)
            gen = torch.Generator(device='cuda').manual_seed(5_100_000+first)
            jitter = scores*torch.exp(.5*torch.randn(scores.shape, device='cuda', generator=gen))
            jitter_order = jitter.argsort(dim=-1, descending=True, stable=True)
            del jitter
            native_order = None
            if trainer is not None:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    base = trainer.model(ids, cache_type='input')
                    ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                    native = torch.cat([ci[n].float()[0] for n in MODULES], -1)
                    native_order = native.argsort(dim=-1, descending=True, stable=True)
                    del base, ci, native
            record = dict(method=name, first_sequence=first, job_id=os.environ['SLURM_JOB_ID'],
                          **metadata, budgets={}, all_kept=assess(model, ids, baseline, check, pieces, torch.ones_like(order, dtype=torch.bool)))
            masks_saved = {}
            for budget in plan['budgets']:
                magnitude = support(order, budget)
                b = dict(magnitude=assess(model, ids, baseline, check, pieces, magnitude), starts={})
                b['magnitude']['fit_quadratic'] = float(.5*residual(z, reference, magnitude).square().mean()/ids.numel())
                best_keep, best_obj = None, float('inf')
                for si, (label, start_order) in enumerate([('magnitude', order), ('jittered_magnitude', jitter_order)]):
                    initial = support(start_order, budget)
                    fitted, log = refine(z, reference, initial, first*100+budget+si, sweeps=args.sweeps)
                    b['starts'][label] = dict(**assess(model, ids, baseline, check, pieces, fitted), search=log,
                                             fit_quadratic=log['objective_history'][-1]/(2*args.fit_k*ids.numel()))
                    if log['objective_history'][-1] < best_obj:
                        best_keep, best_obj = fitted.clone(), log['objective_history'][-1]
                        b['selected_start'] = label
                b['selected'] = b['starts'][b['selected_start']]
                b['overlap_with_magnitude'] = float((best_keep & magnitude).sum()/magnitude.sum())
                masks_saved[str(budget)] = dict(magnitude=magnitude.cpu(), selected=best_keep.cpu())
                if native_order is not None:
                    native_keep = support(native_order, budget)
                    b['native'] = assess(model, ids, baseline, check, pieces, native_keep)
                    b['native']['fit_quadratic'] = float(.5*residual(z, reference, native_keep).square().mean()/ids.numel())
                    masks_saved[str(budget)]['native'] = native_keep.cpu()
                record['budgets'][str(budget)] = b
                print(json.dumps(dict(method=name, sequence=first, budget=budget,
                                      magnitude_kl=b['magnitude']['kl'], selected_kl=b['selected']['kl'],
                                      fit=b['selected']['fit_quadratic'], independent=b['selected']['independent_quadratic'])), flush=True)
            support_path = OUT/f'{first}_{name}_supports.npz'
            np.savez_compressed(support_path, **{
                f'{budget}_{label}': mask.nonzero()[:, 1].reshape(512, int(budget)).numpy().astype(np.uint16)
                for budget, variants in masks_saved.items() for label, mask in variants.items()})
            record['support_sha256'] = sha(support_path)
            record['elapsed_job_seconds'] = time.perf_counter()-began
            save_json(record, dest)
            del pieces, trainer, z, reference, scores, order, jitter_order, native_order, best_keep, fitted
            torch.cuda.empty_cache()
        del cache, check, baseline
    from joint_select67_report import finish
    finish(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
