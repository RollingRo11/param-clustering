"""Historical source-position-local proxy versus measured KLKeep.

WARNING: the field called ``full`` below omits cross-source-position terms.
It is not the full-sequence Fisher prediction. For VPD, this legacy predictor
also omits the delta between the target and the all-component reconstruction.
Use gaussian67_diagnose.py's exact FP32 full-sequence JVP for that comparison.
The historical computation and output fields remain unchanged for reproducibility.

Ranking: Fisher K=8 (seed A), exactly as evaluate_checkpoint. Prediction: a fresh K=8 (seed B).
With z[m,c,k] = (r_c.h)(g_k.u_c), three first-order predictions of KL for removed set S:
  diag    ½ Σ_m Σ_{c∈S} mean_k z²          (per-piece scores only, no cross terms)
  within  ½ Σ_m mean_k (Σ_{c∈S_m} z)²      (cross terms inside each matrix)
  full    ½ mean_k (Σ_m Σ_{c∈S_m} z)²      (cross terms across all 24 matrices)
mean_k (g_k.δ)² estimates Var_p[Δlogit], and KL ≈ ½ Var_p[Δlogit] for small shifts.
The actual KL uses the same masked forwards as KLKeep on the same tokens.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method
import parauto_structured as structured
from fisher_attribution import capture_fisher


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto')
    ap.add_argument('--sequences', type=int, default=128); ap.add_argument('--expected-step', type=int, default=5000); ap.add_argument('--k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['completed'] == args.expected_step and saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    provenance = saved.get('provenance'); model = target()
    if args.method == 'vpd':
        # Same rank-one shape: piece c = v_c u_c^T. Rank by VPD's CI and mask inside its component model,
        # as KLKeep does; predict from the intact target's cache with VPD's own read/write vectors.
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
    fractions = [.0003, .001, .003, .01, .03, .1, .2, .4]
    sums = {f: dict(actual=0., diag=0., within=0., full=0.) for f in fractions}; n_tok = 0; started = time.perf_counter()
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        pred_cache = capture_fisher(model, rows, k=args.k, seed=10_000+first)
        rank_cache = None if args.method == 'vpd' else capture_fisher(model, rows, k=args.k, seed=first)   # needs grad; keep outside no_grad
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if args.method == 'vpd':
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n].float() for n in MODULES]; del base, ci
            else:
                scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(rank_cache, pieces)]; del rank_cache
            parts = [structured.signed_parts(e, p) for e, p in zip(pred_cache, pieces)]
            parts = [(r.float(), s.float()) for r, s in parts]
        del pred_cache
        with torch.no_grad():
            counts = [s.shape[-1] for s in scores]
            order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True); del scores
            with torch.autocast('cuda', dtype=torch.bfloat16): baseline = model(ids)[0]
            logp = baseline.float().log_softmax(-1); prob = logp.exp(); del baseline
            for f in fractions:
                k = round(f*sum(counts)); keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
                masks = keep.split(counts, -1)
                diag = within = full = None
                for (read, sens), mask in zip(parts, masks):
                    rem = (1-mask).reshape(-1, mask.shape[-1])                     # [n,C]
                    z = read[:, None, :]*sens                                       # [n,K,C]
                    d = (z.square().mean(1)*rem).sum(-1)                            # [n]
                    w = torch.einsum('nc,nkc->nk', rem, z)                          # [n,K]
                    diag = d if diag is None else diag+d
                    within = w.square().mean(1) if within is None else within+w.square().mean(1)
                    full = w if full is None else full+w
                full = full.square().mean(1)
                if args.method == 'vpd':
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, masks))))
                else:
                    hooks = []
                    try:
                        for name, p, mask in zip(MODULES, pieces, masks):
                            def replace(layer, inputs, output, p=p, mask=mask):
                                return ((inputs[0] @ p.read)*mask) @ p.write.T
                            hooks.append(model.get_submodule(name).register_forward_hook(replace))
                        with torch.autocast('cuda', dtype=torch.bfloat16): pred = model(ids)[0]
                    finally:
                        for h in hooks: h.remove()
                kl = (prob*(logp-pred.float().log_softmax(-1))).sum(-1).flatten()     # [n]
                s = sums[f]; s['actual'] += float(kl.sum()); s['diag'] += .5*float(diag.sum()); s['within'] += .5*float(within.sum()); s['full'] += .5*float(full.sum())
                del pred, kl, masks, keep
            n_tok += ids.numel()
        print(json.dumps(dict(sequences=first+2)), flush=True)
    rows = [dict(keep_fraction=f, **{k: v/n_tok for k, v in s.items()}) for f, s in sums.items()]
    for r in rows: print(json.dumps(r), flush=True)
    save_json(dict(method=args.method, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), provenance=provenance, positions=n_tok, k=args.k,
                   rank_seed='first', predict_seed='10000+first', rows=rows, seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(),
                   data_sha256=manifest['val']['sha256']), args.output)
    print('JOINT_DIAGNOSTIC_DONE', flush=True)


if __name__ == '__main__': main()
