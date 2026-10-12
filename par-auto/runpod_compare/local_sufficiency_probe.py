"""Is a sparse support layer-locally sufficient? No sparse forward pass is used.

For support S chosen on the intact model, each matrix's own output is

    y   = W h                                     intact
    y_S = sum_{c in S} (r_c . h) u_c  (+ Delta h) kept set only; VPD's dense remainder is never masked

and the layer-local error is the change after that layer's OWN nonlinearity:

    mlp.c_fc      GELU (tanh approximation, as in the target's NewGELU)
    q_proj/k_proj the causal attention probabilities of that layer, rotary applied, exactly as the model
    v/o/down      identity (their write into the residual stream is linear)

Small local errors everywhere mean the sparse model's activations track the intact model's layer by
layer, so an intact-point sensor is enough to train against (the induction argument). Large local
errors with small end-to-end KL mean the support relies on later layers repairing the damage.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method

SPECS = {
    'vpd_published': ('vpd', 400000, 'checkpoints/published_vs_finished/vpd.pth'),
    'vpd_5000': ('vpd', 5000, 'checkpoints/step_5000/vpd.pth'),
    'gim_5000': ('parauto', 5000, 'output/gim_1h/checkpoints/005000/model.pth'),
    'gim_7783': ('parauto', 7783, 'output/gim_1h/checkpoints/007783/model.pth'),
    'fisher_k4_5000': ('parauto', 5000, 'output/fisher_k4_5000/checkpoints/005000/model.pth'),
    'local_avg_5000': ('parauto', 5000, 'output/local_avg_c1_5000/checkpoints/005000/model.pth'),
    'density_b0.5_5000': ('parauto', 5000, 'output/density_b0.5_5000/checkpoints/005000/model.pth'),
}
KIND = {'q_proj': 'attention', 'k_proj': 'attention', 'v_proj': 'linear',
        'o_proj': 'linear', 'c_fc': 'gelu', 'down_proj': 'linear'}


def gelu(x):
    return F.gelu(x, approximate='tanh')


def load(model, name):
    kind, step, rel = SPECS[name]
    path = ROOT/rel
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert saved['method'] == kind and saved['completed'] == step and saved['target_sha256'] == TARGET_SHA
    trainer = None
    if kind == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True)
        handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors'])
        trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(),
                                  trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda().float(), p['write'].cuda().float()) for p in saved['factors']]
    return pieces, trainer, dict(checkpoint=str(path), checkpoint_sha256=sha(path), kind=kind, step=step)


def attention_probabilities(attn, q_flat, k_flat, batch, length):
    """The module's own probabilities, recomputed from (possibly masked) q/k projection outputs."""
    q = q_flat.view(batch, length, attn.n_head, attn.head_dim).transpose(1, 2)
    k = k_flat.view(batch, length, attn.n_key_value_heads, attn.head_dim).transpose(1, 2)
    position_ids = torch.arange(length, device=q.device).unsqueeze(0).clamp(max=attn.n_ctx - 1)
    cos = attn.rotary_cos[position_ids].to(q.dtype)
    sin = attn.rotary_sin[position_ids].to(q.dtype)
    q, k = attn.apply_rotary_pos_emb(q, k, cos, sin)
    if attn.use_grouped_query_attention and attn.repeat_kv_heads > 1:
        k = k.repeat_interleave(attn.repeat_kv_heads, dim=1)
    att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
    att = att.masked_fill(attn.bias[:, :, :length, :length] == 0, float('-inf'))
    return F.softmax(att, dim=-1)


def relative(error, reference):
    """Per-token relative size: ||error|| / ||reference||, both over the feature axis."""
    return (error.float().norm(dim=-1) / reference.float().norm(dim=-1).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', choices=sorted(SPECS), required=True)
    ap.add_argument('--support', choices=['native', 'count'], default='count')
    ap.add_argument('--count', type=float, default=193., help='mean kept pieces per token for --support count')
    ap.add_argument('--sequences', type=int, default=8)
    ap.add_argument('--first-row', type=int, default=472)
    ap.add_argument('--fisher-k', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    started = time.perf_counter()
    model = target()
    pieces, trainer, meta = load(model, args.name)
    weights = [model.get_submodule(n).weight.detach().float() for n in MODULES]
    # VPD's components do not sum to W; its remainder rides along unmasked, exactly as in its own forward.
    deltas = [w - p.write @ p.read.T for w, p in zip(weights, pieces)]
    delta_share = [float(d.norm()/w.norm()) for d, w in zip(deltas, weights)]
    data, _ = tokens('val')
    counts = [p.count for p in pieces]
    acc = {n: {k: [] for k in ('local', 'linear', 'kept')} for n in MODULES}
    kept_total, batches, exactness = [], 0, None

    for first in range(args.first_row, args.first_row + args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda')
        ids = rows[:, :-1]
        batch, length = ids.shape
        if meta['kind'] == 'vpd':
            from write_attribution import capture_inputs
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n].float().flatten(0, 1) for n in MODULES]
                del base, ci
            hs = [e['h'].float() for e in capture_inputs(model, rows)]
        else:
            from fisher_attribution import capture_fisher
            cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores = [method.scores_of(e, p).float() for e, p in zip(cache, pieces)]
            hs = [e['h'].float() for e in cache]
            del cache
        with torch.no_grad():
            if args.support == 'native':
                assert meta['kind'] == 'vpd', 'native support means VPD causal importance > 0'
                masks = [(s > 0).float() for s in scores]
            else:
                order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)
                k = max(1, round(args.count))
                keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
                masks = list(keep.split(counts, -1))
                del order, keep
            del scores
            kept_total.append(torch.stack([m.sum(-1) for m in masks]).sum(0))

            outputs = {}
            for name, h, p, w, delta, mask in zip(MODULES, hs, pieces, weights, deltas, masks):
                y = h @ w.T
                kept = ((h @ p.read) * mask) @ p.write.T + h @ delta.T
                outputs[name] = (y, kept)
                acc[name]['linear'].append(relative(kept - y, y))
                acc[name]['kept'].append(mask.sum(-1))
                if KIND[name.split('.')[-1]] == 'gelu':
                    acc[name]['local'].append(relative(gelu(kept) - gelu(y), gelu(y)))
                elif KIND[name.split('.')[-1]] == 'linear':
                    acc[name]['local'].append(acc[name]['linear'][-1])
            for layer in range(4):
                attn = model.get_submodule(f'h.{layer}.attn')
                q_true, q_kept = outputs[f'h.{layer}.attn.q_proj']
                k_true, k_kept = outputs[f'h.{layer}.attn.k_proj']
                true = attention_probabilities(attn, q_true, k_true, batch, length)
                for name, q, kk in ((f'h.{layer}.attn.q_proj', q_kept, k_true),
                                    (f'h.{layer}.attn.k_proj', q_true, k_kept)):
                    got = attention_probabilities(attn, q, kk, batch, length)
                    # Total-variation distance per (head, query row), averaged over heads, per token.
                    tv = (got - true).abs().sum(-1).mul(.5).mean(1).flatten()
                    acc[name]['local'].append(tv)
                del true
            if exactness is None:       # all-ones support must reproduce every matrix output exactly
                full = []
                for name, h, p, w, delta in zip(MODULES, hs, pieces, weights, deltas):
                    y = h @ w.T
                    full.append(float(relative(((h @ p.read)) @ p.write.T + h @ delta.T - y, y).max()))
                exactness = dict(max_relative_error_with_all_pieces=max(full))
                assert exactness['max_relative_error_with_all_pieces'] < 1e-3, exactness
            del outputs, hs, masks
        batches += 1

    def summary(values):
        v = torch.cat(values).float()
        return dict(mean=float(v.mean()), median=float(v.median()), p90=float(v.quantile(.9)))

    per_matrix = {n: dict(kind=KIND[n.split('.')[-1]], local=summary(acc[n]['local']),
                          linear=summary(acc[n]['linear']), kept_mean=float(torch.cat(acc[n]['kept']).mean()))
                  for n in MODULES}
    by_kind = {}
    for kind in ('linear', 'gelu', 'attention'):
        members = [n for n in MODULES if KIND[n.split('.')[-1]] == kind]
        by_kind[kind] = dict(mean_local=float(np.mean([per_matrix[n]['local']['mean'] for n in members])),
                             worst_matrix=max(members, key=lambda n: per_matrix[n]['local']['mean']))
    result = dict(name=args.name, support=args.support, requested_count=args.count,
                  mean_kept_per_token=float(torch.cat(kept_total).mean()), sequences=args.sequences,
                  first_row=args.first_row, fisher_k=args.fisher_k, batches=batches,
                  provenance=meta, delta_share=dict(zip(MODULES, delta_share)), exactness=exactness,
                  per_matrix=per_matrix, by_kind=by_kind, seconds=round(time.perf_counter()-started, 1),
                  source_sha256=sha(__file__),
                  note=('local error = relative change of each matrix output after its own nonlinearity; '
                        'attention rows report total-variation distance, not a relative norm'))
    save_json(result, args.output)
    print(json.dumps({k: result[k] for k in ('name', 'support', 'mean_kept_per_token', 'by_kind', 'seconds')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
