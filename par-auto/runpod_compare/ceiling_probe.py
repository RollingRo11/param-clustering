"""Can k pieces per matrix reproduce that matrix's output at a token? Four levels of difficulty.

Per matrix, per held-out token, relative error ||W h - approximation|| / ||W h|| for k = 1, 2, 4, 8:

  ours      the method's own ranking (Fisher) keeps its top k of this matrix, coefficients a_c = r_c.h
  oracle    greedy best k pieces, same coefficients a_c: the most any ranking could buy
  refit     greedy best k pieces, least-squares coefficients: what the linear encoder costs
  svd       best rank-k approximation of the matrix's output across this batch of tokens, one shared
            basis (not per token): a floor that any fixed dictionary with per-token selection faces

Greedy step: pick the piece minimizing the residual, i.e. maximizing 2 a_c <resid, u_c> - a_c^2 ||u_c||^2
(for refit, the usual orthogonal-matching-pursuit gain <resid, u_c>^2 / ||u_c||^2).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json


def greedy(true, a, U, ks, refit):
    """true [n,d]; a [n,C] coefficients; U [d,C] writes. Returns {k: relative error [n]}."""
    n, d = true.shape
    resid = true.clone()
    chosen = torch.zeros(n, 0, dtype=torch.long, device=true.device)
    norms = U.square().sum(0)                                             # [C]
    out, picked = {}, torch.zeros(n, U.shape[1], dtype=torch.bool, device=true.device)
    for step in range(1, max(ks)+1):
        dot = resid @ U                                                   # [n,C]
        gain = dot.square()/norms.clamp_min(1e-30) if refit else 2*a*dot - a.square()*norms
        gain = gain.masked_fill(picked, -float('inf'))
        best = gain.argmax(-1)                                            # [n]
        picked.scatter_(1, best[:, None], True)
        chosen = torch.cat((chosen, best[:, None]), 1)
        if refit:                                                         # least squares on the chosen set
            sub = U[:, chosen].permute(1, 0, 2)                           # [n,d,step]
            coef = torch.linalg.lstsq(sub.float(), true[:, :, None].float()).solution
            approx = (sub @ coef).squeeze(-1)
        else:
            approx = (a.gather(1, chosen)[:, None, :]*U[:, chosen].permute(1, 0, 2)).sum(-1)
        resid = true - approx
        if step in ks: out[step] = resid.norm(dim=-1)/true.norm(dim=-1).clamp_min(1e-12)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sequences', type=int, default=8); ap.add_argument('--tokens-per-batch', type=int, default=128); ap.add_argument('--fisher-k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    from train_pair import core, method
    from fisher_attribution import capture_fisher
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'parauto' and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'))
    pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]; del saved
    model = target(); weights = [model.get_submodule(n).weight.detach().float() for n in MODULES]
    data, manifest = tokens('val')
    ks = [1, 2, 4, 8]
    acc = {n: {lv: {k: [] for k in ks} for lv in ('ours', 'oracle', 'refit', 'svd')} for n in MODULES}
    started = time.perf_counter()
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            scores = [method.scores_of(e, p).float() for e, p in zip(cache, pieces)]
        hs = [e['h'].float() for e in cache]; del cache
        index = torch.randperm(len(hs[0]), device='cuda', generator=torch.Generator(device='cuda').manual_seed(first))[:args.tokens_per_batch]
        with torch.no_grad():
            for name, h, p, w, s in zip(MODULES, hs, pieces, weights, scores):
                h, s = h[index], s[index]
                a = h @ p.read                                            # [n,C]
                U = p.write.float()                                       # [d,C]
                true = h @ w.T
                order = s.argsort(-1, descending=True)
                for k in ks:                                              # the method's own ranking
                    sel = order[:, :k]
                    approx = (a.gather(1, sel)[:, None, :]*U[:, sel].permute(1, 0, 2)).sum(-1)
                    acc[name]['ours'][k].append((true-approx).norm(dim=-1)/true.norm(dim=-1).clamp_min(1e-12))
                for level, refit in (('oracle', False), ('refit', True)):
                    for k, v in greedy(true, a, U, ks, refit).items(): acc[name][level][k].append(v)
                basis = torch.linalg.svd(true, full_matrices=False)       # shared rank-k floor on this batch
                for k in ks:
                    approx = (basis.U[:, :k]*basis.S[:k]) @ basis.Vh[:k]
                    acc[name]['svd'][k].append((true-approx).norm(dim=-1)/true.norm(dim=-1).clamp_min(1e-12))
        print(json.dumps(dict(sequences=first+2, seconds=round(time.perf_counter()-started, 1))), flush=True)
    summary = {n: {lv: {str(k): float(torch.cat(v).median()) for k, v in d.items()} for lv, d in m.items()} for n, m in acc.items()}
    overall = {lv: {str(k): float(np.median([summary[n][lv][str(k)] for n in MODULES])) for k in ks} for lv in ('ours', 'oracle', 'refit', 'svd')}
    print(json.dumps(dict(median_relative_error_over_matrices=overall), indent=1), flush=True)
    save_json(dict(checkpoint=str(args.checkpoint), **meta, tokens=args.sequences//2*args.tokens_per_batch,
                   gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'],
                   overall=overall, per_matrix=summary, seconds=time.perf_counter()-started), args.output)
    print('CEILING_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
