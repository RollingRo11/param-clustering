"""Are the kept components faithful parts of the computation, or large cancelling artifacts?

At one keep fraction (default 1%), each method ranked natively (VPD: CI; par-auto: --sensor),
same held-out tokens, one GPU:

A. write mass      sum_c |r_c.h| ||u_c|| / ||W h||, over all pieces and over the kept set.
   kept output     ||sum_kept u_c (r_c.h)|| / ||W h||, and its cosine with the true W h.
   A large write mass with a small resulting vector means the pieces are big and cancel.
   A kept set with cosine ~1 and ratio ~1 reproduces that matrix's actual output.

B. masked run      cosine and norm ratio of the masked vs intact residual stream at each block
   input and before the final norm. Shows whether damage is a rescaling (which RMSNorm undoes),
   a direction change, or is corrected downstream.

C. mask transplant KL with each token's own mask, with another token's mask (positions permuted),
   and with a random mask of the same size. If a transplanted mask costs no more than the token's
   own, the per-token selection carries no token-specific mechanism.

All three read the intact target's activations; only B and C run masked forwards.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method


def stats(values):
    v = torch.cat(values).float()
    q = torch.quantile(v[torch.randperm(len(v), device=v.device)[:1_000_000]], torch.tensor([.1, .5, .9], device=v.device))
    return dict(mean=float(v.mean()), p10=float(q[0]), median=float(q[1]), p90=float(q[2]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto'); ap.add_argument('--sensor', choices=['gim', 'fisher'], default='fisher')
    ap.add_argument('--sequences', type=int, default=64); ap.add_argument('--keep', type=float, default=.01); ap.add_argument('--fisher-k', type=int, default=8)
    ap.add_argument('--random-init', action='store_true', help='replace pieces with seed-0 projected random factors (same counts): the untrained baseline')
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'))
    model = target()
    if args.method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        from param_decomp.models.components import make_mask_infos
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    del saved
    weights = [model.get_submodule(n).weight.detach().float() for n in MODULES]
    if args.random_init:
        assert args.method == 'parauto'
        pieces = [core.initialize_pieces(w, p.read.shape[1], 0) for w, p in zip(weights, pieces)]
        meta['note'] = 'random projected factors at the trained run\'s component counts; no training'
    write_norms = [p.write.float().square().sum(0).sqrt() for p in pieces]                 # [C]
    data, manifest = tokens('val')
    blocks = [f'block{i}' for i in range(4)]+['final']
    acc = {k: [[] for _ in MODULES] for k in ('mass_all', 'mass_kept', 'kept_ratio', 'kept_cosine', 'cancel_kept')}
    resid = {b: dict(cosine=[], norm_ratio=[]) for b in blocks}
    kl = dict(own=0., transplant=0., random=0.); n_tok = 0; started = time.perf_counter()

    def capture_residual(run):
        seen, hooks = {}, []
        def grab(tag):
            def hook(_m, inputs, _out): seen[tag] = inputs[0].detach().float()
            return hook
        for i in range(4): hooks.append(model.h[i].register_forward_hook(grab(f'block{i}')))
        hooks.append(model.ln_f.register_forward_hook(grab('final')))
        try: out = run()
        finally:
            for h in hooks: h.remove()
        assert len(seen) == 5, seen.keys()
        return out, seen

    def masked_forward(ids, masks):
        if args.method == 'vpd':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, masks))))
        hooks = []
        try:
            for name, p, mask in zip(MODULES, pieces, masks):
                def replace(layer, inputs, output, p=p, mask=mask):
                    return ((inputs[0] @ p.read)*mask) @ p.write.T
                hooks.append(model.get_submodule(name).register_forward_hook(replace))
            with torch.autocast('cuda', dtype=torch.bfloat16): return model(ids)[0]
        finally:
            for h in hooks: h.remove()

    generator = torch.Generator(device='cuda').manual_seed(0)
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        if args.method == 'parauto':
            if args.sensor == 'gim':
                from gim_attribution import capture_gim
                cache = capture_gim(model, rows)
            else:
                from fisher_attribution import capture_fisher
                cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
            hs = [e['h'].float() for e in cache]; del cache
        else:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n].float() for n in MODULES]; del base, ci
            from write_attribution import capture_inputs
            hs = [e['h'].float() for e in capture_inputs(model, rows)]
        with torch.no_grad():
            counts = [s.shape[-1] for s in scores]
            order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True); del scores
            k = round(args.keep*sum(counts))
            keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.); del order
            masks = list(keep.split(counts, -1))
            # A. write mass and how much of the matrix output the kept set reproduces
            for i, (h, p, w, norms, mask) in enumerate(zip(hs, pieces, weights, write_norms, masks)):
                a = h @ p.read                                                              # [n,C]
                flat = mask.reshape(-1, counts[i])
                true = h @ w.T
                true_norm = true.norm(dim=-1).clamp_min(1e-12)
                kept_out = (a*flat) @ p.write.T
                kept_norm = kept_out.norm(dim=-1)
                acc['mass_all'][i].append((a.abs()*norms).sum(-1)/true_norm)
                acc['mass_kept'][i].append((a.abs()*norms*flat).sum(-1)/true_norm)
                acc['kept_ratio'][i].append(kept_norm/true_norm)
                acc['kept_cosine'][i].append(torch.nn.functional.cosine_similarity(kept_out, true, dim=-1))
                acc['cancel_kept'][i].append((a.abs()*norms*flat).sum(-1)/kept_norm.clamp_min(1e-12))
                del a, true, kept_out
            # B. residual stream under the mask, against the intact run
            with torch.autocast('cuda', dtype=torch.bfloat16):
                intact, intact_res = capture_residual(lambda: model(ids)[0])
            masked, masked_res = capture_residual(lambda: masked_forward(ids, masks))
            for b in blocks:
                x, y = intact_res[b].flatten(0, 1), masked_res[b].flatten(0, 1)
                resid[b]['cosine'].append(torch.nn.functional.cosine_similarity(x, y, dim=-1))
                resid[b]['norm_ratio'].append(y.norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-12))
            logp = intact.float().log_softmax(-1); prob = logp.exp()
            divergence = lambda pred: float((prob*(logp-pred.float().log_softmax(-1))).sum())
            kl['own'] += divergence(masked); del masked, intact, intact_res, masked_res
            # C. another token's mask, and a random mask of the same size
            perm = torch.randperm(keep.shape[0]*keep.shape[1], device='cuda', generator=generator)
            shuffled = [m.reshape(-1, c)[perm].reshape(m.shape) for m, c in zip(masks, counts)]
            kl['transplant'] += divergence(masked_forward(ids, shuffled)); del shuffled
            noise = torch.rand(keep.shape, device='cuda', generator=generator)
            rnd = torch.zeros_like(keep).scatter_(-1, noise.argsort(dim=-1)[..., :k], 1.)
            kl['random'] += divergence(masked_forward(ids, list(rnd.split(counts, -1)))); del rnd, noise, keep, masks
            n_tok += ids.numel()
        print(json.dumps(dict(sequences=first+2, seconds=round(time.perf_counter()-started, 1))), flush=True)
    result = dict(method=args.method, sensor=args.sensor if args.method == 'parauto' else 'CI', keep_fraction=args.keep,
                  checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), random_init=args.random_init, **meta, positions=n_tok,
                  gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'],
                  per_matrix={n: {k: stats(acc[k][i]) for k in acc} for i, n in enumerate(MODULES)},
                  overall={k: stats([v for row in acc[k] for v in row]) for k in acc},
                  residual={b: {k: stats(v) for k, v in d.items()} for b, d in resid.items()},
                  kl={k: v/n_tok for k, v in kl.items()}, seconds=time.perf_counter()-started)
    for k in ('mass_all', 'mass_kept', 'kept_ratio', 'kept_cosine', 'cancel_kept'):
        print(json.dumps({k: {m: round(v, 4) for m, v in result['overall'][k].items()}}), flush=True)
    print(json.dumps(dict(residual={b: dict(cosine=round(result['residual'][b]['cosine']['median'], 4),
                                            norm_ratio=round(result['residual'][b]['norm_ratio']['median'], 4)) for b in blocks})), flush=True)
    print(json.dumps(dict(kl={k: round(v, 4) for k, v in result['kl'].items()})), flush=True)
    save_json(result, args.output)
    print('FAITHFULNESS_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
