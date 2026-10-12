"""Frozen 67M diagnostics: ranking variance, local errors, layer interactions, exact JVP.

No fitting. Native VPD excludes delta, matching the published CI evaluation.
JVP is the derivative of the full sequence intervention, including temporal cross terms.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['PAIR_ROOT'])
SNAP = ROOT/'code/gaussian67_20260914'
sys.path.insert(0, str(SNAP/'compare'))
import numpy as np
import torch
from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha, upload
from train_pair import VPD, core, method
from fisher_attribution import capture_fisher
from gaussian_attribution import trace

SPECS = {
    'gim': ('parauto', 5000, 'output/gim_1h/checkpoints/005000/model.pth'),
    'fisher': ('parauto', 5000, 'output/fisher_k4_5000/checkpoints/005000/model.pth'),
    'gaussian': ('parauto', 5000, 'output/gaussian67_20260914/gaussian/checkpoints/005000/model.pth'),
    'boundary001': ('parauto', 5000, 'output/gaussian67_20260914/boundary001/checkpoints/005000/model.pth'),
    'vpd_5000': ('vpd', 5000, 'checkpoints/step_5000/vpd.pth'),
    'vpd_published': ('vpd', 400000, 'checkpoints/published_vs_finished/vpd.pth'),
}


def load_models(model):
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    result = {}
    for name, (kind, step, rel) in SPECS.items():
        path = ROOT/rel
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert saved['method'] == kind and saved['completed'] == step and saved['target_sha256'] == TARGET_SHA
        trainer = None
        if kind == 'vpd':
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
        result[name] = dict(pieces=pieces, trainer=trainer, metadata=dict(checkpoint=str(path), sha256=sha(path),
                            step=step, matrix_reconstruction_errors=errors))
        del saved
    return result


@contextmanager
def replacements(model, pieces, masks, *, alpha=1., layers=(0, 1, 2, 3), modules=None, deltas=None):
    hooks = []
    try:
        for m, (name, p, mask) in enumerate(zip(MODULES, pieces, masks)):
            if int(name.split('.')[1]) not in layers or (modules is not None and m not in modules): continue
            def hook(layer, inputs, out, m=m, p=p, mask=mask):
                if deltas is not None: return out + alpha*deltas[m].to(out.dtype)
                value = ((inputs[0] @ p.read)*mask.to(inputs[0].dtype)) @ p.write.T
                if layer.bias is not None: value = value+layer.bias
                if isinstance(alpha, float) and alpha == 1.: return value
                return out + alpha*(value-out)
            hooks.append(model.get_submodule(name).register_forward_hook(hook))
        yield
    finally:
        for hook in hooks: hook.remove()


def forward(model, ids, pieces, masks, *, fp32=False, **kw):
    with replacements(model, pieces, masks, **kw), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not fp32):
        return model(ids)[0]


def kl(logits, reference):
    logp = reference.float().log_softmax(-1)
    return (logp.exp()*(logp-logits.float().log_softmax(-1))).sum(-1).mean(-1)


@torch.no_grad()
def scores_both(cache, pieces, shape):
    out8, out64 = [], []
    for e, p in zip(cache, pieces):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            r = (e['h'] @ p.read).float().abs()
            sumsq = torch.zeros_like(r)
            for start in range(0, 64, 8):
                effect = (e['events']['cot'][:, start:start+8] @ p.write).float()
                sumsq += effect.square().sum(1)
                if start == 0: out8.append((r*(sumsq/8).sqrt()).reshape(*shape, -1))
            out64.append((r*(sumsq/64).sqrt()).reshape(*shape, -1))
    return out8, out64


def order_of(scores):
    return torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)


def mask_of(order, counts, budget):
    return list(torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :budget], 1.).split(counts, -1))


def seq_mean(x):
    return x.detach().float().mean(-1).cpu().tolist()


def exact_path(model, ids, pieces, masks):
    """FP32 scalar-parameter JVP includes all matrices and all source positions."""
    fn = lambda alpha: forward(model, ids, pieces, masks, alpha=alpha, fp32=True)
    with torch.enable_grad():
        base, tangent = torch.autograd.functional.jvp(fn, torch.zeros((), device='cuda'),
                            torch.ones((), device='cuda'), create_graph=False, strict=False)
    with torch.no_grad():
        logp = base.float().log_softmax(-1); prob = logp.exp()
        quadratic = .5*((prob*tangent.square()).sum(-1)-(prob*tangent).sum(-1).square()).mean(-1)
        result = dict(quadratic_kl=quadratic.tolist(), path={})
        for alpha in [.001, .01, .1, .25, .5, 1.]:
            out = fn(alpha)
            result['path'][str(alpha)] = kl(out, base).tolist()
            if alpha == .001:
                fd = (out-base)/alpha
                result['finite_difference_relative_error'] = ((fd-tangent).flatten(1).norm(dim=1)/tangent.flatten(1).norm(dim=1)).tolist()
        result['base_parity'] = float((base-model(ids)[0]).abs().max())
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sequences', type=int, default=16); ap.add_argument('--start', type=int, default=384)
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Real-model diagnostics must run under SLURM'
    credentials(); torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    from param_decomp.models.components import make_mask_infos
    init_distributed()
    started = time.perf_counter(); model = target(); models = load_models(model)
    data, manifest = tokens('val')
    assert 0 <= args.start and args.start+args.sequences <= len(data) and args.sequences % 2 == 0
    result = dict(target_sha256=TARGET_SHA, parameters=sum(p.numel() for p in model.parameters()),
                  data_sha256=manifest['val']['sha256'], start_sequence=args.start, sequences=args.sequences,
                  job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
                  source_sha256=sha(__file__), ranking='intact sampled-label Fisher, nested K=8 / K=64; native CI separately',
                  jvp_note='FP32 exact full-sequence tangent; first two contexts only; masks fixed from BF16 ranking',
                  methods={n: dict(**v['metadata'], batches=[]) for n, v in models.items()})
    for first in range(args.start, args.start+args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids=rows[:, :-1]
        cache = capture_fisher(model, rows, k=64, seed=910_000+first, microbatch=2)
        with torch.no_grad(), trace(model, require_grad=False) as (inputs, outputs, _):
            with torch.autocast('cuda', dtype=torch.bfloat16): baseline = model(ids)[0]
            inputs = [inputs[n].detach().float() for n in MODULES]
            outputs = [outputs[n].detach().float() for n in MODULES]
        for name, v in models.items():
            pieces, trainer = v['pieces'], v['trainer']; counts=[p.count for p in pieces]
            s8, s64 = scores_both(cache, pieces, ids.shape)
            with torch.no_grad():
                orders = dict(fisher8=order_of(s8), fisher64=order_of(s64))
                b = dict(first_sequence=first, budgets={}, all_kept={})
                if trainer is not None:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        base = trainer.model(ids, cache_type='input')
                        ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                        native = [ci[n].float() for n in MODULES]
                    orders['native'] = order_of(native)
                    b['native_positive_count'] = seq_mean(torch.cat(native, -1).gt(0).sum(-1))
                    b['baseline_parity'] = float((base.output-baseline).abs().max())
                    del base, ci, native
                main_rank = 'native' if trainer else 'fisher64'
                all_masks = [torch.ones_like(s) for s in s64]
                b['all_kept']['bf16'] = kl(forward(model, ids, pieces, all_masks), baseline).tolist()
                if first == args.start:
                    base32 = model(ids)[0]
                    b['all_kept']['fp32'] = kl(forward(model, ids, pieces, all_masks, fp32=True), base32).tolist()
                    del base32
                combined = torch.cat(s64, -1)
                b['score_participation_count'] = seq_mean(combined.sum(-1).square()/combined.square().sum(-1).clamp_min(1e-30))
                for budget in [512, 4096]:
                    masks = {k: mask_of(o, counts, budget) for k, o in orders.items()}
                    main_masks = masks[main_rank]
                    d = dict(actual={}, main_rank=main_rank)
                    for rank, parts in masks.items():
                        d['actual'][rank] = kl(forward(model, ids, pieces, parts), baseline).tolist()
                    d['overlap_8_64'] = seq_mean(sum((a*c).sum(-1) for a, c in zip(masks['fisher8'], masks['fisher64']))/budget)
                    d['retained_score_mass64'] = seq_mean(sum((s*m).sum(-1) for s,m in zip(s64, main_masks))/combined.sum(-1).clamp_min(1e-30))
                    d['retained_score_energy64'] = seq_mean(sum((s.square()*m).sum(-1) for s,m in zip(s64, main_masks))/combined.square().sum(-1).clamp_min(1e-30))
                    d['matrix_kept'] = [seq_mean(m.sum(-1)) for m in main_masks]
                    deltas=[]; d['local_relative_rms']=[]
                    for m, (p, mask) in enumerate(zip(pieces, main_masks)):
                        value = ((inputs[m] @ p.read)*mask) @ p.write.T
                        bias = model.get_submodule(MODULES[m]).bias
                        if bias is not None: value += bias
                        delta=value-outputs[m]; deltas.append(delta)
                        d['local_relative_rms'].append((delta.square().sum((1,2))/outputs[m].square().sum((1,2)).clamp_min(1e-30)).sqrt().tolist())
                    d['frozen_local_errors_actual'] = kl(forward(model, ids, pieces, main_masks, deltas=deltas), baseline).tolist()
                    d['only_layer'] = {str(l): kl(forward(model, ids, pieces, main_masks, layers=(l,)), baseline).tolist() for l in range(4)}
                    d['restore_layer'] = {str(l): kl(forward(model, ids, pieces, main_masks, layers=tuple(j for j in range(4) if j!=l)), baseline).tolist() for l in range(4)}
                    if budget == 512:
                        d['only_matrix'] = {n: kl(forward(model, ids, pieces, main_masks, modules=(m,)), baseline).tolist() for m,n in enumerate(MODULES)}
                    if trainer and first == args.start:
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            native_out = trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, main_masks))))
                        hook_out = forward(model, ids, pieces, main_masks)
                        d['vpd_hook_max_difference'] = float((native_out-hook_out).abs().max())
                        assert d['vpd_hook_max_difference'] < .1
                    if first == args.start:
                        d['exact_path'] = exact_path(model, ids, pieces, main_masks)
                    b['budgets'][str(budget)] = d
                result['methods'][name]['batches'].append(b)
                del s8, s64, orders, combined, masks, main_masks, all_masks, deltas
            result['seconds'] = time.perf_counter()-started
            save_json(result, args.output/'diagnostic.json')
            print(json.dumps(dict(method=name, first_sequence=first, seconds=round(result['seconds'], 1))), flush=True)
        del cache, baseline, inputs, outputs
    result['complete']=True; save_json(result, args.output/'diagnostic.json')
    receipts=[upload(args.output/'diagnostic.json', 'experiments/gaussian67_diagnostic_20260914/diagnostic.json'),
              upload(Path(__file__), 'experiments/gaussian67_diagnostic_20260914/gaussian67_diagnose.py')]
    save_json(receipts,args.output/'r2_receipts.json')
    print('GAUSSIAN67_DIAGNOSTIC_DONE', flush=True)


if __name__ == '__main__': main()
