"""Per-token computational graph of each decomposition, and how many edges it takes to explain the output.

Nodes: the top-N components at a token by the method's native importance (VPD: CI; par-auto: Fisher).
Edges: the first-order path a component's write actually takes to a later component's read.

  o_proj / down_proj (write the residual) -> q,k,v,c_fc (read it, later in the network)
      edge = a_c * (r_c' . (rms_weight * u_c)) / rms          RMS scale frozen, as in GIM
  c_fc -> down_proj (same MLP, through the GELU)
      edge = a_c * (r_c' . (gelu'(z) * u_c))

Attention-internal edges (q,k,v -> o_proj, which cross positions through the softmax) are not included;
both methods lose the same edge class.

Edge attribution = |edge| * (that downstream node's own effect on the output distribution, RMS_k(g_k.u)),
i.e. how much of the downstream node's output effect this edge supplies. Both methods are scored with the
same Fisher sensitivity, so edge mass is in the same units.

Reported per token: edges needed for 80% of edge attribution, in-degree needed for 80% of a node's incoming
edge mass, how much of each node's activation the in-graph edges account for, and what share of the whole
token's output effect the N nodes carry.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha

LAYERS = 4
KIND = [n.split('.')[-1] for n in MODULES]
LAYER_OF = [int(n.split('.')[1]) for n in MODULES]


def edges_for(i, j):
    """Is matrix i a first-order writer into matrix j's input, and by which path?"""
    li, lj = LAYER_OF[i], LAYER_OF[j]
    if KIND[i] == 'c_fc' and KIND[j] == 'down_proj' and li == lj: return 'gelu'
    if KIND[i] not in ('o_proj', 'down_proj') or KIND[j] not in ('q_proj', 'k_proj', 'v_proj', 'c_fc'): return None
    if lj > li: return 'rms_1' if KIND[j] != 'c_fc' else 'rms_2'
    if lj == li and KIND[i] == 'o_proj' and KIND[j] == 'c_fc': return 'rms_2'   # attention writes before rms_2
    return None


def count_for_share(values, share=.8):
    """How many of the largest entries reach `share` of the total (per row)."""
    if values.numel() == 0: return None
    ordered = values.sort(descending=True).values
    total = ordered.sum().clamp_min(1e-30)
    return int((ordered.cumsum(0)/total < share).sum().item())+1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto'); ap.add_argument('--nodes', type=int, default=256)
    ap.add_argument('--sequences', type=int, default=8); ap.add_argument('--tokens-per-batch', type=int, default=32); ap.add_argument('--fisher-k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    from train_pair import core, method
    from fisher_attribution import capture_fisher
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'))
    model = target()
    if args.method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    del saved
    norm_weight = {f'{i}.rms_{k}': getattr(model.h[i], f'rms_{k}').weight.detach().float() for i in range(LAYERS) for k in (1, 2)}
    eps = model.h[0].rms_1.variance_epsilon
    data, manifest = tokens('val')
    pairs = [(i, j, p) for i in range(len(MODULES)) for j in range(len(MODULES)) if (p := edges_for(i, j))]
    stats = {k: [] for k in ('edges_for_80', 'in_degree_80', 'coverage', 'node_effect_share', 'edges_nonzero', 'out_degree_80')}
    started = time.perf_counter()
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
        resid, hidden, hooks = {}, {}, []
        def grab_norm(tag):
            def hook(_m, inputs, _out): resid[tag] = inputs[0].detach().float().flatten(0, 1)
            return hook
        def grab_z(tag):
            def hook(_m, _i, out): hidden[tag] = out.detach().float().flatten(0, 1)
            return hook
        for i in range(LAYERS):
            for k in (1, 2): hooks.append(getattr(model.h[i], f'rms_{k}').register_forward_hook(grab_norm(f'{i}.rms_{k}')))
            hooks.append(model.get_submodule(f'h.{i}.mlp.c_fc').register_forward_hook(grab_z(f'{i}.z')))
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): model(ids)
        finally:
            for h in hooks: h.remove()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            sens = [torch.linalg.vector_norm((e['events']['cot'].float() @ p.write.float()), dim=1).float()/math.sqrt(args.fisher_k)
                    for e, p in zip(cache, pieces)]                                   # [n,C] output effect per unit activation
            if args.method == 'vpd':
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                native = [ci.lower_leaky[n].float().flatten(0, 1) for n in MODULES]; del base, ci
            else:
                native = [method.scores_of(e, p).float() for e, p in zip(cache, pieces)]
        acts = [(e['h'].float() @ p.read) for e, p in zip(cache, pieces)]              # [n,C]
        del cache
        with torch.no_grad():
            counts = [a.shape[-1] for a in acts]
            fisher_effect = [a.abs()*s for a, s in zip(acts, sens)]                    # comparable units for both
            index = torch.randperm(acts[0].shape[0], device='cuda', generator=torch.Generator(device='cuda').manual_seed(first))[:args.tokens_per_batch]
            combined = torch.cat([n[index] for n in native], -1)
            order = combined.argsort(-1, descending=True)[:, :args.nodes]
            effect_all = torch.cat([f[index] for f in fisher_effect], -1)
            offsets = np.cumsum([0]+counts)
            for t in range(len(index)):
                tok = int(index[t])
                sel = {}
                chosen = order[t].sort().values
                for m in range(len(MODULES)):
                    inside = chosen[(chosen >= offsets[m]) & (chosen < offsets[m+1])]-offsets[m]
                    if len(inside): sel[m] = inside
                stats['node_effect_share'].append(float(effect_all[t].gather(0, order[t]).sum()/effect_all[t].sum().clamp_min(1e-30)))
                edge_attr, in_edges = [], {m: [] for m in sel}
                out_edges = {m: [] for m in sel}
                for i, j, path in pairs:
                    if i not in sel or j not in sel: continue
                    u = pieces[i].write.float()[:, sel[i]]                              # [d,ni]
                    if path == 'gelu':
                        z = hidden[f'{LAYER_OF[i]}.z'][tok]
                        t_ = torch.tanh(math.sqrt(2/math.pi)*(z+.044715*z**3))
                        scale = .5*(1+t_) + .5*z*(1-t_*t_)*math.sqrt(2/math.pi)*(1+3*.044715*z*z)   # gelu'
                        transfer = scale[:, None]*u
                    else:
                        x = resid[f'{LAYER_OF[j]}.{path}'][tok]
                        rms = (x.pow(2).mean()+eps).sqrt()
                        transfer = (norm_weight[f'{LAYER_OF[j]}.{path}'][:, None]*u)/rms
                    block = (pieces[j].read.float()[:, sel[j]].T @ transfer)*acts[i][tok, sel[i]][None, :]   # [nj,ni]
                    edge_attr.append((block.abs()*sens[j][tok, sel[j]][:, None]).flatten())
                    in_edges[j].append(block); out_edges[i].append(block.T)
                if not edge_attr: continue
                attr = torch.cat(edge_attr)
                stats['edges_for_80'].append(count_for_share(attr))
                stats['edges_nonzero'].append(int((attr > 1e-12).sum()))
                for m, blocks in in_edges.items():
                    if not blocks: continue
                    got = torch.cat(blocks, -1)                                        # [nj, upstream]
                    for r in range(got.shape[0]):
                        stats['in_degree_80'].append(count_for_share(got[r].abs()))
                        stats['coverage'].append(float(got[r].sum().abs()/acts[m][tok, sel[m][r]].abs().clamp_min(1e-12)))
                for m, blocks in out_edges.items():
                    if not blocks: continue
                    got = torch.cat(blocks, -1)
                    for r in range(got.shape[0]): stats['out_degree_80'].append(count_for_share(got[r].abs()))
        print(json.dumps(dict(sequences=first+2, seconds=round(time.perf_counter()-started, 1))), flush=True)
    summary = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)), n=len(v)) for k, v in stats.items() if v}
    print(json.dumps(summary, indent=1), flush=True)
    save_json(dict(method=args.method, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta,
                   nodes=args.nodes, tokens=len(stats['node_effect_share']), edge_paths=len(pairs),
                   gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'], summary=summary,
                   seconds=time.perf_counter()-started), args.output)
    print('GRAPH_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
