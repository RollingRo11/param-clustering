"""Adversarial masking, as in the VPD paper's Sec 3.2 table, applied identically to VPD and par-auto.

Mask per component, per position: m = g + (1 - g) * r, r in [0, 1].
  g  = the gate: VPD's own causal importance (--gates native, VPD only), or a 0/1 top-k keep set ranked by
       the method's score (--gates topk; VPD: CI, par-auto: Fisher). g = 1 components cannot be touched.
  r  = adversary. Random init, then sign-gradient ascent on KL(target || masked), step 0.1, clamp to [0, 1].
       --scope position: one r per position;  --scope shared: one r per component per microbatch;
       --scope global: one r per component for ALL --sequences (gradients summed over microbatches). This is
       VPD's eval: param-decomp pile_llama_simple_mlp-4L.yaml PGDReconLoss_{20..320}step, source_shape c,
       eval batch 128 x 512, step 0.1, init random (published-run code: mask_scope shared_across_batch).
KL is recorded after 0, 20, 40, 80, 160, 320 steps of one trajectory (the loop is deterministic, so the
20-step value is exactly what a separate 20-step run gives). Also recorded: r = 0 (hard ablation, KLkeep).
VPD's delta component is excluded, as in the paper's table. par-auto pieces sum to W exactly.
Paper (published VPD, native gates, 128 x 512 tokens): 20:0.83 40:1.35 80:3.84 160:25.3 320:40.2.
--adversary hard: the model sees each dropped piece fully on or fully off (on where r > 0.5), never in between, for
decompositions trained with a hard forward; the gradient passes straight through to r, which climbs as above.
--init off starts every r at 0.5 - step/2 (all dropped pieces off: the KLKeep point; one up-step switches a piece on).
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
# Old cluster snapshots hold only this file and import the rest from code/compare; a git checkout has it all beside it.
if (ROOT/'code/compare').is_dir() and not (Path(__file__).resolve().parent/'train_pair.py').exists():
    sys.path.insert(0, str(ROOT/'code/compare'))
import numpy as np
import torch
from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto'); ap.add_argument('--gates', default='topk')
    ap.add_argument('--keeps', default='0.005,0.01,0.1'); ap.add_argument('--scope', choices=['position', 'shared', 'global'], default='position')
    ap.add_argument('--sequences', type=int, default=32); ap.add_argument('--microbatch', type=int, default=8)
    ap.add_argument('--record', default='0,20,40,80,160,320'); ap.add_argument('--step-size', type=float, default=.1)
    ap.add_argument('--fisher-k', type=int, default=8)
    ap.add_argument('--sensor', choices=['fisher', 'write', 'static'], default='fisher')   # par-auto ranking; write = |r.h| ||u||, static = |r.h| sqrt(u^T M u)
    ap.add_argument('--adversary', choices=['soft', 'hard'], default='soft'); ap.add_argument('--init', choices=['random', 'off'], default='random')
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    leak = (lambda r: (r > .5).float() + r - r.detach()) if args.adversary == 'hard' else (lambda r: r)
    def start(shape, generator):
        if args.init == 'off': return torch.full(shape, .5 - args.step_size / 2, device='cuda').requires_grad_(True)
        return torch.rand(shape, device='cuda', generator=generator).requires_grad_(True)
    record = sorted(int(s) for s in args.record.split(',')); keeps = [float(k) for k in args.keeps.split(',')]
    gate_kinds = args.gates.split(',')
    assert args.method == 'vpd' or gate_kinds == ['topk']
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
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    # static sensor: the run's fixed output metric (static_fisher.py), saved in every model.pth
    metric = [(f['L'].cuda(), f['eigenvalues']) for f in saved['static_metric']] if 'static_metric' in saved else None
    del saved
    data, manifest = tokens('val')
    configs = (['native'] if 'native' in gate_kinds else [])+([f'top{k:g}' for k in keeps] if 'topk' in gate_kinds else [])
    sums = {c: {**{str(s): 0. for s in record}, 'hard_zero': 0.} for c in configs}; n_tok = 0; started = time.perf_counter()

    def forward(ids, masks):
        if args.method == 'vpd':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, masks)))).float()
        hooks = []
        try:
            for name, p, mask in zip(MODULES, pieces, masks):
                def replace(_layer, a, _o, p=p, mask=mask):
                    return ((a[0] @ p.read)*mask) @ p.write.T
                hooks.append(model.get_submodule(name).register_forward_hook(replace))
            with torch.autocast('cuda', dtype=torch.bfloat16): return model(ids)[0].float()
        finally:
            for h in hooks: h.remove()

    def prepare(first):
        """ids, intact logits (bf16), and per-config gates for one microbatch."""
        rows = torch.tensor(np.array(data[first:first+args.microbatch], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        if args.method == 'parauto':
            if args.sensor == 'write':
                from write_attribution import capture_inputs
                cache = capture_inputs(model, rows)
            elif args.sensor == 'static':
                import static_fisher
                from write_attribution import capture_inputs
                cache = static_fisher.entries(capture_inputs(model, rows), metric)
            else:
                from fisher_attribution import capture_fisher
                cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
            del cache
        else:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                scores = [ci[n].float() for n in MODULES]; del base, ci
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(ids)[0].bfloat16()
            counts = [s.shape[-1] for s in scores]
            order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)
            gates = {}
            for c in configs:
                if c == 'native': gates[c] = [s.clamp(0, 1).half() for s in scores]
                else:
                    k = round(float(c[3:])*sum(counts))
                    gates[c] = list(torch.zeros(order.shape, dtype=torch.bool, device='cuda').scatter_(-1, order[..., :k], True).split(counts, -1))
        return ids, logits, gates

    def kl_sum(logits, out):
        logp = logits.float().log_softmax(-1)
        return (logp.exp()*(logp-out.log_softmax(-1))).sum()

    if args.scope == 'global':
        batches = [prepare(first) for first in range(0, args.sequences, args.microbatch)]
        n_tok = sum(ids.numel() for ids, _, _ in batches)
        for c in configs:
            shapes = [g.shape[-1] for g in batches[0][2][c]]
            with torch.no_grad():
                for ids, logits, gates in batches: sums[c]['hard_zero'] += float(kl_sum(logits, forward(ids, [g.float() for g in gates[c]])))
            generator = torch.Generator(device='cuda').manual_seed(0)
            adv = [start((C,), generator) for C in shapes]
            for step in range(record[-1]+1):
                total, grads = 0., [torch.zeros_like(r) for r in adv]
                for ids, logits, gates in batches:
                    loss = kl_sum(logits, forward(ids, [g.float()+(1-g.float())*leak(r) for g, r in zip(gates[c], adv)]))
                    total += float(loss.detach())
                    if step < record[-1]:
                        for acc, gr in zip(grads, torch.autograd.grad(loss/n_tok, adv)): acc += gr
                    del loss
                if step in record: sums[c][str(step)] = total
                if step == record[-1]: break
                with torch.no_grad():
                    for r, gr in zip(adv, grads): r.add_(args.step_size*gr.sign()).clamp_(0., 1.)
                if step in record or step % 20 == 0:
                    print(json.dumps(dict(config=c, step=step, kl=round(total/n_tok, 4), seconds=round(time.perf_counter()-started, 1))), flush=True)
            print(json.dumps(dict(config=c, done={s: round(v/n_tok, 4) for s, v in sums[c].items()})), flush=True)
            del adv
    else:
        for first in range(0, args.sequences, args.microbatch):
            ids, logits, gates_all = prepare(first)
            for c in configs:
                gates = [g.float() for g in gates_all[c]]
                with torch.no_grad(): sums[c]['hard_zero'] += float(kl_sum(logits, forward(ids, gates)))
                generator = torch.Generator(device='cuda').manual_seed(first)
                shape = lambda g: g.shape if args.scope == 'position' else (1, 1, g.shape[-1])
                adv = [start(shape(g), generator) for g in gates]
                for step in range(record[-1]+1):
                    loss = kl_sum(logits, forward(ids, [g+(1-g)*leak(r) for g, r in zip(gates, adv)]))
                    if step in record: sums[c][str(step)] += float(loss.detach())
                    if step == record[-1]: break
                    grads = torch.autograd.grad(loss/ids.numel(), adv)
                    with torch.no_grad():
                        for r, gr in zip(adv, grads): r.add_(args.step_size*gr.sign()).clamp_(0., 1.)
                    del loss, grads
                del adv, gates
            n_tok += ids.numel(); del gates_all, logits
            print(json.dumps(dict(sequences=first+args.microbatch, seconds=round(time.perf_counter()-started, 1),
                                  running={c: {s: round(v/n_tok, 3) for s, v in d.items()} for c, d in sums.items()})), flush=True)
    result = {c: {s: v/n_tok for s, v in d.items()} for c, d in sums.items()}
    save_json(dict(method=args.method, ranking=('CI' if args.method == 'vpd' else args.sensor), scope=args.scope, step_size=args.step_size,
                   adversary=args.adversary, init=args.init,
                   checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta, positions=n_tok, sequences=args.sequences,
                   microbatch=args.microbatch, gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'],
                   kl=result, seconds=time.perf_counter()-started), args.output)
    print(json.dumps(result), flush=True)
    print('ADVERSARIAL_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
