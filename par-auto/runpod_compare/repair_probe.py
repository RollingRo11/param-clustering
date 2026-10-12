"""Does damage in early layers get erased only when later layers are masked? Many damage types, many layer splits.

For every split of the 4 layers into an upstream set U and a later downstream set D (max U < min D; 17 pairs):

  damage on U (5 kinds)
    vpd          VPD's own keep mask (global top-k by CI) on U's matrices
    vpd_random   VPD pieces, same number kept per matrix per token, chosen at random
    parauto      par-auto's keep mask (global top-k by Fisher) on U's matrices
    noise_matrix Gaussian noise added to each U matrix output, per token sized like VPD's mask error there
    noise_resid  Gaussian noise added to the residual entering the first layer after U, per token sized like
                 the residual error VPD's U mask causes at that point
  downstream D (5 modes)
    frozen       D's residual writes (o_proj, down_proj) fixed to their intact values: no response at all
    real         the real weights: the model's own response (its self-repair, if any)
    vpd          VPD's keep mask on D;  vpd_random: random VPD mask of the same size;  parauto: par-auto's mask
Added damage A = KL(U damaged, D mode) - KL(D mode, no U damage). Real and frozen have no damage alone.
If A(vpd) << A(real), masking D erases upstream damage that the real later layers let through.

All replacements are forward hooks on the target, so every combination runs in one network. VPD pieces are its
V, U without the delta component (checked against VPD's own masked forward at start).
"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np
import torch
from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method

LAYER_OF = [int(n.split('.')[1]) for n in MODULES]
WRITER = [n.endswith(('o_proj', 'down_proj')) for n in MODULES]
DAMAGES = ('vpd', 'vpd_random', 'parauto', 'noise_matrix', 'noise_resid')
MODES = ('frozen', 'real', 'vpd', 'vpd_random', 'parauto')
name = lambda s: '+'.join(map(str, s))


def pairs():
    out = []
    for s in range(3):
        for extra in itertools.chain.from_iterable(itertools.combinations(range(s), r) for r in range(s+1)):
            U = tuple(extra)+(s,)
            for r in range(1, 4-s):
                for D in itertools.combinations(range(s+1, 4), r): out.append((U, D))
    return out


def global_keep(scores, f):
    counts = [s.shape[-1] for s in scores]
    order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)
    k = round(f*sum(counts))
    return list(torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.).split(counts, -1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--vpd', type=Path, required=True); ap.add_argument('--parauto', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--sequences', type=int, default=64)
    ap.add_argument('--keeps', default='0.01,0.1'); ap.add_argument('--fisher-k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    keeps = [float(k) for k in args.keeps.split(',')]
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    model = target()
    from train_pair import VPD
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    from param_decomp.models.components import make_mask_infos
    saved = torch.load(args.vpd, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'vpd' and saved['target_sha256'] == TARGET_SHA
    vpd_meta = dict(completed=saved.get('completed'), sha256=sha(args.vpd))
    trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
    trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False); del saved
    vpd = [core.FreePieces(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().float().T) for n in MODULES]
    saved = torch.load(args.parauto, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'parauto' and saved['target_sha256'] == TARGET_SHA
    par_meta = dict(completed=saved.get('completed'), sha256=sha(args.parauto))
    par = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]; del saved
    data, manifest = tokens('val')
    PAIRS = pairs(); assert len(PAIRS) == 17
    Us = sorted({U for U, _ in PAIRS}); Ds = sorted({D for _, D in PAIRS})
    kl = {f: dict(real={d: {name(U): [] for U in Us} for d in DAMAGES},
                  pair={d: {name(U): {name(D): {m: [] for m in MODES if m != 'real'} for D in Ds} for U in Us} for d in DAMAGES},
                  alone={m: {name(D): [] for D in Ds} for m in ('vpd', 'vpd_random', 'parauto')}) for f in keeps}
    scale_log = {f: dict(noise_matrix_rel=[], noise_resid_rel=[]) for f in keeps}
    check = None; started = time.perf_counter()

    def run(ids, matrix_fns, pre_fns=None, capture=False):
        """matrix_fns: {module index: fn(x, out) -> out}; pre_fns: {block: fn(x) -> x}."""
        hooks, seen = [], {}
        try:
            for m, fn in matrix_fns.items():
                hooks.append(model.get_submodule(MODULES[m]).register_forward_hook(lambda _l, a, o, fn=fn: fn(a[0], o)))
            for b, fn in (pre_fns or {}).items():
                hooks.append(model.h[b].register_forward_pre_hook(lambda _l, a, fn=fn: (fn(a[0]), *a[1:])))
            if capture:
                for b in (1, 2, 3):
                    hooks.append(model.h[b].register_forward_pre_hook(lambda _l, a, b=b: seen.__setitem__(b, a[0].detach().float())))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): out = model(ids)[0].float()
        finally:
            for h in hooks: h.remove()
        return (out, seen) if capture else out

    mask_fn = lambda p, keep: (lambda x, o: ((x @ p.read)*keep) @ p.write.T)
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        from fisher_attribution import capture_fisher
        cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            par_scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, par)]
        del cache
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            base = trainer.model(ids, cache_type='input')
            ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
            vpd_scores = [ci[n].float() for n in MODULES]; del base, ci
        # intact run: matrix inputs and outputs, block inputs, target distribution
        inputs, outputs, hooks = {}, {}, []
        for m, n in enumerate(MODULES):
            def grab(_l, a, o, m=m): inputs[m] = a[0].detach(); outputs[m] = o.detach()
            hooks.append(model.get_submodule(n).register_forward_hook(grab))
        try: intact, intact_res = run(ids, {}, capture=True)
        finally:
            for h in hooks: h.remove()
        logp = intact.log_softmax(-1); prob = logp.exp(); del intact
        seq_kl = lambda out: (prob*(logp-out.log_softmax(-1))).sum(-1).mean(-1).tolist()      # per sequence
        generator = torch.Generator(device='cuda').manual_seed(first)
        for f in keeps:
            K = kl[f]
            keep_v = global_keep(vpd_scores, f); keep_p = global_keep(par_scores, f)
            keep_r = []
            for k in keep_v:                                                         # same count per matrix per token
                rank = torch.rand(k.shape, device='cuda', generator=generator).argsort(-1).argsort(-1)
                keep_r.append((rank < k.sum(-1, keepdim=True)).float())
            if check is None:                                                        # hooks == VPD's own masked forward
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    theirs = trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, keep_v)))).float()
                mine = run(ids, {m: mask_fn(vpd[m], keep_v[m]) for m in range(len(MODULES))})
                check = dict(kl_vpd_forward=float(np.mean(seq_kl(theirs))), kl_hooks=float(np.mean(seq_kl(mine))),
                             max_logit_diff=float((theirs-mine).abs().max()))
                print(json.dumps(dict(equivalence_check=check)), flush=True); del theirs, mine
                assert abs(check['kl_vpd_forward']-check['kl_hooks']) < 1e-2*max(check['kl_vpd_forward'], 1e-3)+1e-4, check
            directions = {m: torch.randn(outputs[m].shape, device='cuda', generator=generator) for m in range(len(MODULES))}
            directions = {m: d/d.norm(dim=-1, keepdim=True) for m, d in directions.items()}
            resid_dir = {b: torch.randn(intact_res[b].shape, device='cuda', generator=generator) for b in (1, 2, 3)}
            resid_dir = {b: d/d.norm(dim=-1, keepdim=True) for b, d in resid_dir.items()}
            with torch.no_grad():                                                    # VPD mask's own error size, per token
                rel = {}
                for m in range(len(MODULES)):
                    true = outputs[m].float()
                    approx = ((inputs[m].float() @ vpd[m].read)*keep_v[m]) @ vpd[m].write.T
                    rel[m] = ((approx-true).norm(dim=-1)/true.norm(dim=-1).clamp_min(1e-12))
            layer_mods = lambda layers: [m for m in range(len(MODULES)) if LAYER_OF[m] in layers]
            down_fns = dict(vpd=lambda D: {m: mask_fn(vpd[m], keep_v[m]) for m in layer_mods(D)},
                            vpd_random=lambda D: {m: mask_fn(vpd[m], keep_r[m]) for m in layer_mods(D)},
                            parauto=lambda D: {m: mask_fn(par[m], keep_p[m]) for m in layer_mods(D)},
                            frozen=lambda D: {m: (lambda x, o, m=m: outputs[m]) for m in layer_mods(D) if WRITER[m]})
            for D in Ds:
                for mode in ('vpd', 'vpd_random', 'parauto'): K['alone'][mode][name(D)] += seq_kl(run(ids, down_fns[mode](D)))
            for U in Us:
                s = max(U)
                _, vpd_res = run(ids, {m: mask_fn(vpd[m], keep_v[m]) for m in layer_mods(U)}, capture=True)
                resid_norm = (vpd_res[s+1]-intact_res[s+1]).norm(dim=-1, keepdim=True); del vpd_res
                scale_log[f]['noise_resid_rel'].append(float((resid_norm.squeeze(-1)/intact_res[s+1].norm(dim=-1)).median()))
                scale_log[f]['noise_matrix_rel'].append(float(torch.stack([rel[m].median() for m in layer_mods(U)]).median()))
                damage = dict(
                    vpd=({m: mask_fn(vpd[m], keep_v[m]) for m in layer_mods(U)}, {}),
                    vpd_random=({m: mask_fn(vpd[m], keep_r[m]) for m in layer_mods(U)}, {}),
                    parauto=({m: mask_fn(par[m], keep_p[m]) for m in layer_mods(U)}, {}),
                    noise_matrix=({m: (lambda x, o, m=m: o+(rel[m][..., None]*o.float().norm(dim=-1, keepdim=True)*directions[m]).to(o.dtype))
                                   for m in layer_mods(U)}, {}),
                    noise_resid=({}, {s+1: (lambda x, n=resid_norm, d=resid_dir[s+1]: x+(n*d).to(x.dtype))}))
                for dname, (fns, pre) in damage.items():
                    K['real'][dname][name(U)] += seq_kl(run(ids, fns, pre))
                    for D in [D for U_, D in PAIRS if U_ == U]:
                        for mode in ('frozen', 'vpd', 'vpd_random', 'parauto'):
                            K['pair'][dname][name(U)][name(D)][mode] += seq_kl(run(ids, {**fns, **down_fns[mode](D)}, pre))
            del keep_v, keep_p, keep_r, directions, resid_dir, rel
        del inputs, outputs, intact_res, logp, prob
        f0 = keeps[0]; K = kl[f0]; mean = lambda v: round(float(np.mean(v)), 3)
        print(json.dumps(dict(sequences=first+2, seconds=round(time.perf_counter()-started, 1),
                              check_U012_D3={d: dict(real=mean(K['real'][d]['0+1+2']), frozen=mean(K['pair'][d]['0+1+2']['3']['frozen']),
                                                     vpd=mean(K['pair'][d]['0+1+2']['3']['vpd']), vpd_alone=mean(K['alone']['vpd']['3']),
                                                     parauto=mean(K['pair'][d]['0+1+2']['3']['parauto']), parauto_alone=mean(K['alone']['parauto']['3']))
                                             for d in DAMAGES})), flush=True)
    save_json(dict(vpd=str(args.vpd), vpd_meta=vpd_meta, parauto=str(args.parauto), parauto_meta=par_meta, sequences=args.sequences,
                   positions_per_sequence=512, pairs=[dict(U=list(U), D=list(D)) for U, D in PAIRS], equivalence_check=check,
                   noise_scale={str(f): {k: float(np.median(v)) for k, v in d.items()} for f, d in scale_log.items()},
                   gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'],
                   kl={str(f): v for f, v in kl.items()}, seconds=time.perf_counter()-started), args.output)
    print('REPAIR_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
