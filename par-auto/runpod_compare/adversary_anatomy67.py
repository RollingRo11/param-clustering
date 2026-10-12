"""Why does VPD's adversarial eval break our decomposition but not VPD, and why did the piece-mass penalty not change
that? Three measurements on the eval's 128 held-out sequences (val rows 0-127, 8 per microbatch), one checkpoint:

1. Cancellation on real text. Per matrix, sum_t sum_c ||v_c(x_t)||^2 over sum_t ||W x_t||^2, with v_c(x) = (r_c . x) u_c
   (VPD: act_c U_c) at the intact input x: the 'activation mass', the real-text version of the weight mass
   sum_c ||u_c||^2 ||r_c||^2 / ||W||_F^2 (the same ratio for isotropic random inputs). 1 = no cancellation.
2. What the dropped pieces carry. At each keep fraction (the top k per position by the method's own score, as the
   eval: par-auto Fisher K=8, VPD causal importance): the masked output error sqrt(sum_t ||W x_t - kept_t||^2) and the
   dropped pieces' output energy sqrt(sum_t sum_dropped ||v_c||^2), both over sqrt(sum_t ||W x_t||^2), per matrix.
3. The attack. VPD's eval adversary exactly as adversarial_probe.py --scope global --sequences 128 (one r per piece,
   random init with cuda seed 0, sign ascent step 0.1 on the summed KL, clamp [0, 1]), stopped after 20 steps. Then:
   the share of pieces it turned on (r > 0.5); the KL when only the N pieces with the largest first-order share of the
   attack (r_c times dKL/dr_c at the final r) keep their r and every other piece gets r = 0; and the KL when its r is
   applied in one layer only, or one matrix type only.
KL as in the eval: sum over positions of KL(p_intact || p_masked) / positions, bf16 logits, fp32 softmax.
Usage: adversary_anatomy67.py --method parauto|vpd --checkpoint PATH --output JSON [--keeps 0.01,0.1]"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch

from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method

SEQUENCES, MICROBATCH, STEPS, STEP_SIZE, FISHER_K = 128, 8, 20, 0.1, 8
TOP_N = (1, 10, 100, 1000, 10000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['parauto', 'vpd'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--keeps', default='0.01,0.1')
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    started = time.perf_counter()
    vpd = args.method == 'vpd'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'))
    model = target()
    if vpd:
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        from param_decomp.models.components import make_mask_infos
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        # (read, write) per matrix: its masked output is ((x @ read) * mask) @ write.T; weight [d_out, d_in].
        factors = [(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().T.float()) for n in MODULES]
        targets = [trainer.model.target_weight(n).detach().float() for n in MODULES]
    else:
        from fisher_attribution import capture_fisher
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
        factors = [(p.read.float(), p.write.float()) for p in pieces]
        targets = []
        for n in MODULES:
            m = model.get_submodule(n)
            targets.append((m.weight.T if type(m).__name__ == 'RadfordConv1D' else m.weight).detach().float())
    del saved
    for (read, write), w in zip(factors, targets): assert w.shape == (write.shape[0], read.shape[0])
    counts = [read.shape[1] for read, _ in factors]
    total = sum(counts)
    keeps = [float(k) for k in args.keeps.split(',')]
    configs = [f'top{k:g}' for k in keeps]
    data, manifest = tokens('val')
    wnorm2 = [write.pow(2).sum(0) for _, write in factors]                        # ||u_c||^2 per piece
    weight_mass = [float((read.pow(2).sum(0) * w2).sum() / w.pow(2).sum()) for (read, _), w2, w in zip(factors, wnorm2, targets)]

    def forward(ids, masks):
        if vpd:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, masks)))).float()
        hooks = []
        try:
            for name, p, mask in zip(MODULES, pieces, masks):
                hooks.append(model.get_submodule(name).register_forward_hook(
                    lambda _layer, a, _o, p=p, mask=mask: ((a[0] @ p.read) * mask) @ p.write.T))
            with torch.autocast('cuda', dtype=torch.bfloat16): return model(ids)[0].float()
        finally:
            for h in hooks: h.remove()

    def kl_sum(logits, out):
        logp = logits.float().log_softmax(-1)
        return (logp.exp() * (logp - out.log_softmax(-1))).sum()

    # 1-2: per microbatch, the scores, gates and intact inputs; the per-matrix sums are taken here, inputs are not kept.
    sums = dict(target=torch.zeros(len(MODULES), dtype=torch.float64), energy=torch.zeros(len(MODULES), dtype=torch.float64),
                **{f'{c}/{key}': torch.zeros(len(MODULES), dtype=torch.float64) for c in configs for key in ('error', 'dropped')})
    batches = []
    for first in range(0, SEQUENCES, MICROBATCH):
        rows = torch.tensor(np.array(data[first:first+MICROBATCH], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        if vpd:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                scores = [ci[n].float() for n in MODULES]
                inputs = [base.cache[n].flatten(0, 1).float() for n in MODULES]
            del base, ci
        else:
            cache = capture_fisher(model, rows, k=FISHER_K, seed=first)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
            inputs = [e['h'].float() for e in cache]
            del cache
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(ids)[0].bfloat16()
            order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)
            gates = {}
            for c, keep in zip(configs, keeps):
                k = round(keep * total)
                gates[c] = list(torch.zeros(order.shape, dtype=torch.bool, device='cuda').scatter_(-1, order[..., :k], True).split(counts, -1))
            del order, scores
            for i, ((read, write), w, x) in enumerate(zip(factors, targets, inputs)):
                acts = x @ read                                                      # [T, C]
                y = x @ w.T
                sums['target'][i] += float(y.double().pow(2).sum())
                energy = acts.pow(2) * wnorm2[i]                                     # ||v_c||^2 per position and piece
                sums['energy'][i] += float(energy.double().sum())
                for c in configs:
                    g = gates[c][i].flatten(0, 1)
                    sums[f'{c}/error'][i] += float(((acts * g) @ write.T - y).double().pow(2).sum())
                    sums[f'{c}/dropped'][i] += float((energy * ~g).double().sum())
                del acts, y, energy
            del inputs
        batches.append((ids, logits, gates))
        print(json.dumps(dict(prepared=first + MICROBATCH, seconds=round(time.perf_counter() - started))), flush=True)
    n_tok = sum(ids.numel() for ids, _, _ in batches)
    per_matrix = [dict(module=n, components=counts[i], weight_mass=weight_mass[i],
                       activation_mass=float(sums['energy'][i] / sums['target'][i]),
                       **{f'{c}/masked_error': float((sums[f'{c}/error'][i] / sums['target'][i]).sqrt()) for c in configs},
                       **{f'{c}/dropped_energy': float((sums[f'{c}/dropped'][i] / sums['target'][i]).sqrt()) for c in configs})
                  for i, n in enumerate(MODULES)]
    totals = dict(weight_mass=float(sum(m * float(w.pow(2).sum()) for m, w in zip(weight_mass, targets)) / sum(float(w.pow(2).sum()) for w in targets)),
                  activation_mass=float(sums['energy'].sum() / sums['target'].sum()))

    # 3: the attack.
    def kl_with(c, rs):
        with torch.no_grad():
            return sum(float(kl_sum(logits, forward(ids, [g.float() + (1 - g.float()) * r for g, r in zip(gates[c], rs)])))
                       for ids, logits, gates in batches) / n_tok

    layer_of = torch.cat([torch.full((C,), int(n.split('.')[1])) for n, C in zip(MODULES, counts)]).cuda()
    kinds = list(dict.fromkeys(n.split('.', 2)[2] for n in MODULES))
    kind_of = torch.cat([torch.full((C,), kinds.index(n.split('.', 2)[2])) for n, C in zip(MODULES, counts)]).cuda()
    attack = {}
    for c in configs:
        generator = torch.Generator(device='cuda').manual_seed(0)
        adv = [torch.rand((C,), device='cuda', generator=generator).requires_grad_(True) for C in counts]
        trace = []
        for step in range(STEPS + 1):
            value, grads = 0., [torch.zeros_like(r) for r in adv]
            for ids, logits, gates in batches:
                loss = kl_sum(logits, forward(ids, [g.float() + (1 - g.float()) * r for g, r in zip(gates[c], adv)]))
                value += float(loss.detach())
                for acc, gr in zip(grads, torch.autograd.grad(loss / n_tok, adv)): acc += gr
                del loss
            trace.append(value / n_tok)
            if step == STEPS: break                                  # the final pass only gives the gradient at r_20
            with torch.no_grad():
                for r, gr in zip(adv, grads): r.add_(STEP_SIZE * gr.sign()).clamp_(0., 1.)
        r = torch.cat([a.detach() for a in adv]); grad = torch.cat(grads)
        share = r * grad                                              # first-order part of the attack's KL per piece
        ranked = share.argsort(descending=True)
        top_n = {}
        for n in TOP_N + (total,):
            keep_r = torch.zeros_like(r); keep_r[ranked[:n]] = r[ranked[:n]]
            top_n[str(n)] = kl_with(c, keep_r.split(counts))
        by_layer = {str(L): kl_with(c, torch.where(layer_of == L, r, 0.).split(counts)) for L in range(4)}
        by_kind = {kind: kl_with(c, torch.where(kind_of == k, r, 0.).split(counts)) for k, kind in enumerate(kinds)}
        attack[c] = dict(kl_trace=trace, on_share=float((r > .5).float().mean()), full_on_share=float((r >= .95).float().mean()),
                         first_order_positive_share=float((share > 0).float().mean()), kl_top_n=top_n,
                         kl_one_layer=by_layer, kl_one_kind=by_kind, hard_zero=kl_with(c, [torch.zeros(C, device='cuda') for C in counts]))
        print(json.dumps({c: {k: v for k, v in attack[c].items() if k != 'kl_trace'}, 'kl_20': trace[-1]}), flush=True)
        del adv, grads
    save_json(dict(method=args.method, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta, positions=n_tok,
                   sequences=SEQUENCES, data_sha256=manifest['val']['sha256'], ranking='CI' if vpd else f'Fisher K={FISHER_K}',
                   steps=STEPS, step_size=STEP_SIZE, totals=totals, per_matrix=per_matrix, attack=attack,
                   gpu=torch.cuda.get_device_name(), seconds=time.perf_counter() - started), args.output)
    print(json.dumps(dict(totals=totals)), flush=True)
    for m in per_matrix:
        print(f"{m['module']:18s} weight_mass {m['weight_mass']:6.3f} act_mass {m['activation_mass']:7.3f} "
              + ' '.join(f"{c}: err {m[f'{c}/masked_error']:.3f} dropped {m[f'{c}/dropped_energy']:.3f}" for c in configs), flush=True)
    print('ADVERSARY_ANATOMY_DONE', flush=True)


if __name__ == '__main__': main()
