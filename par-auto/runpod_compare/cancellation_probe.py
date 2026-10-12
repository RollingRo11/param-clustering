"""Is VPD's cross-layer cancellation part of the model's computation, or only of its masks?

One per-token keep mask per keep fraction, ranked globally over all 24 matrices by the method's native
score (VPD: CI; par-auto: --sensor). The SAME mask is then applied to every subset S of the 4 layers
(15 non-empty subsets). Layers outside S run the target's real weights. So the only thing that changes
between subsets is which layers are approximated. (perlayer_klkeep.py re-ranked within each layer and
filled unmasked VPD layers with components at mask 1; this removes both differences.)

Recorded per keep fraction and subset:
  kl            KL(intact || masked), summed over positions
  resid_err     ||x_masked - x_intact|| / ||x_intact|| at block 1-3 inputs and before the final norm
Recorded for the four single-layer subsets and the all-layer one:
  final error vectors e_S = x_masked - x_intact before the final norm, for pairwise cosines between
  single-layer errors and ||sum_L e_L|| vs ||e_all||
Recorded for the all-layer run only:
  kept_act_err  per layer, ||a_kept(masked run) - a_kept(intact run)|| / ||a_kept(intact run)||, where
                a_c = r_c . x is a kept component's activation. Near 0: the kept circuit runs on the
                inputs the real model gives it. Large: it runs on different inputs.

Reading:
  - If masking more layers LOWERS KL (e.g. KL(all) < KL({0})), the extra masks repair earlier damage.
    The real model's downstream weights are present in the single-layer run, so the repair is done by
    removing components, not by the model's own downstream computation.
"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method

LAYER_OF = [int(n.split('.')[1]) for n in MODULES]
POINTS = ['block1', 'block2', 'block3', 'final']
SUBSETS = [s for s in range(1, 16)]                                    # bit l set = layer l masked
SINGLES = {1 << l: l for l in range(4)}


def summary(values):
    v = torch.cat(values).float()
    return dict(mean=float(v.mean()), median=float(v.median()), n=len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto'); ap.add_argument('--sensor', choices=['gim', 'fisher'], default='fisher')
    ap.add_argument('--sequences', type=int, default=64); ap.add_argument('--keeps', default='0.003,0.01,0.03,0.1,0.2'); ap.add_argument('--fisher-k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    keeps = [float(k) for k in args.keeps.split(',')]
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
    data, manifest = tokens('val')
    kl = {(f, s): 0. for f in keeps for s in SUBSETS}
    resid = {(f, s, p): [] for f in keeps for s in SUBSETS for p in POINTS}
    pair_cos = {(f, a, b): [] for f in keeps for a, b in itertools.combinations(range(4), 2)}
    err_norm = {(f, name): [] for f in keeps for name in ('sum_of_single_norms', 'norm_of_single_sum', 'all_layers')}
    kept_err = {(f, l): [] for f in keeps for l in range(4)}
    n_tok = 0; started = time.perf_counter()

    def run(ids, masks, want_inputs):
        """masks: {module index: [b,t,C]} for the masked modules only. Returns logits, residuals, matrix inputs."""
        seen, inputs, hooks = {}, {}, []
        def grab(tag):
            def hook(_m, a, _o): seen[tag] = a[0].detach().float().flatten(0, 1)
            return hook
        def grab_input(m):
            def hook(_m, a, _o): inputs[m] = a[0].detach().float().flatten(0, 1)
            return hook
        try:
            for i in (1, 2, 3): hooks.append(model.h[i].register_forward_hook(grab(f'block{i}')))
            hooks.append(model.ln_f.register_forward_hook(grab('final')))
            if want_inputs:
                for m, name in enumerate(MODULES): hooks.append(model.get_submodule(name).register_forward_hook(grab_input(m)))
            if args.method == 'vpd':
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = trainer.model(ids, mask_infos=make_mask_infos({MODULES[m]: k for m, k in masks.items()})) if masks else model(ids)[0]
            else:
                for m, mask in masks.items():
                    def replace(_layer, a, _o, p=pieces[m], mask=mask):
                        return ((a[0] @ p.read)*mask) @ p.write.T
                    hooks.append(model.get_submodule(MODULES[m]).register_forward_hook(replace))
                with torch.autocast('cuda', dtype=torch.bfloat16): out = model(ids)[0]
        finally:
            for h in hooks: h.remove()
        assert len(seen) == 4 and (not want_inputs or len(inputs) == len(MODULES))
        return out.float(), seen, inputs

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
            del cache
        else:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n].float() for n in MODULES]; del base, ci
        with torch.no_grad():
            counts = [s.shape[-1] for s in scores]
            order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True); del scores
            intact, intact_res, intact_in = run(ids, {}, True)
            logp = intact.log_softmax(-1); prob = logp.exp(); del intact
            for f in keeps:
                k = round(f*sum(counts))
                keep = list(torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.).split(counts, -1))
                errors = {}
                for s in SUBSETS:
                    masks = {m: keep[m] for m in range(len(MODULES)) if s >> LAYER_OF[m] & 1}
                    out, res, ins = run(ids, masks, s == 15)
                    kl[(f, s)] += float((prob*(logp-out.log_softmax(-1))).sum()); del out
                    for p in POINTS:
                        resid[(f, s, p)].append(((res[p]-intact_res[p]).norm(dim=-1)/intact_res[p].norm(dim=-1).clamp_min(1e-12)).cpu())
                    if s in SINGLES or s == 15: errors[s] = res['final']-intact_res['final']
                    if s == 15:
                        num = {l: 0. for l in range(4)}; den = {l: 0. for l in range(4)}
                        for m in range(len(MODULES)):
                            flat = keep[m].reshape(-1, counts[m]); read = pieces[m].read.float()
                            base_act = (intact_in[m] @ read)*flat
                            num[LAYER_OF[m]] = num[LAYER_OF[m]]+(((ins[m] @ read)*flat-base_act).square().sum(-1))
                            den[LAYER_OF[m]] = den[LAYER_OF[m]]+base_act.square().sum(-1)
                        for l in range(4): kept_err[(f, l)].append((num[l].sqrt()/den[l].sqrt().clamp_min(1e-12)).cpu())
                    del res, ins
                scale = intact_res['final'].norm(dim=-1).clamp_min(1e-12)
                single = [errors[1 << l] for l in range(4)]
                for a, b in itertools.combinations(range(4), 2):
                    pair_cos[(f, a, b)].append(torch.nn.functional.cosine_similarity(single[a], single[b], dim=-1).cpu())
                err_norm[(f, 'sum_of_single_norms')].append((sum(e.norm(dim=-1) for e in single)/scale).cpu())
                err_norm[(f, 'norm_of_single_sum')].append((sum(single).norm(dim=-1)/scale).cpu())
                err_norm[(f, 'all_layers')].append((errors[15].norm(dim=-1)/scale).cpu())
                del keep, errors, single
            del intact_res, intact_in, order
            n_tok += ids.numel()
        print(json.dumps(dict(sequences=first+2, seconds=round(time.perf_counter()-started, 1))), flush=True)
    name = lambda s: '+'.join(str(l) for l in range(4) if s >> l & 1)
    by_keep = {}
    for f in keeps:
        by_keep[str(f)] = dict(
            kl={name(s): kl[(f, s)]/n_tok for s in SUBSETS},
            resid_err={name(s): {p: summary(resid[(f, s, p)]) for p in POINTS} for s in SUBSETS},
            single_error_cosine={f'{a}-{b}': summary(pair_cos[(f, a, b)]) for a, b in itertools.combinations(range(4), 2)},
            final_error_norms={k: summary(err_norm[(f, k)]) for k in ('sum_of_single_norms', 'norm_of_single_sum', 'all_layers')},
            kept_act_err_all_layers={str(l): summary(kept_err[(f, l)]) for l in range(4)})
        row = by_keep[str(f)]
        print(json.dumps(dict(keep=f, kl={k: round(v, 3) for k, v in row['kl'].items()},
                              kept_act_err={l: round(v['median'], 3) for l, v in row['kept_act_err_all_layers'].items()},
                              final_err={k: round(v['median'], 3) for k, v in row['final_error_norms'].items()})), flush=True)
    save_json(dict(method=args.method, ranking=args.sensor if args.method == 'parauto' else 'CI', checkpoint=str(args.checkpoint),
                   checkpoint_sha256=sha(args.checkpoint), **meta, positions=n_tok, gpu=torch.cuda.get_device_name(),
                   data_sha256=manifest['val']['sha256'], keeps=by_keep, seconds=time.perf_counter()-started), args.output)
    print('CANCELLATION_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
