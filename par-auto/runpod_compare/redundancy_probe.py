"""How much could merging redundant pieces shrink a decomposition, exactly or nearly?

Two pieces merge into ONE rank-one piece without error when
  read merge   their activations are proportional on the data, a_j = alpha a_i  (u_i a_i + u_j a_j = (u_i + alpha u_j) a_i)
  write merge  their writes are parallel, u_j = beta u_i                         (= u_i (r_i + beta r_j) . h)
Similarity used: read  |cos| of activations over tokens, from G = R' Sigma R with Sigma = E[h h'] (uncentered, since a
                       merge needs proportionality, not correlation);  write |cos| of write vectors.
Per matrix and threshold tau in {0.8, 0.9, 0.95, 0.99}: connected components of pairs above tau (read and write
merges done separately, never mixed, so each merged group is still rank one), each group replaced by its reference
member (largest activation or write norm) with least-squares coefficients.
Reported:
  partner share      fraction of pieces with some other piece above tau
  kept share         pieces left / pieces before
  output error       E||sum_c u_c a_c - merged output||^2 / E||W h||^2 on held-out data (exact from G, U'U)
  pieces for 80%     per token, how many pieces (global over 24 matrices) carry 80% of the Fisher score mass,
                     before and after merging (Fisher score |r.h| RMS_k(g_k.u), same units for every method)
Controls: --method vpd (V, U of a VPD checkpoint), --random-init (untrained projected pieces, par-auto counts).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core

TAUS = (0.8, 0.9, 0.95, 0.99)


def components(sim, tau):
    """Connected components of |sim| > tau (diagonal ignored). Returns labels [C] (numpy)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    C = sim.shape[0]
    edges = ((sim > tau) & ~torch.eye(C, dtype=torch.bool, device=sim.device)).nonzero().cpu().numpy()
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(C, C))
    return connected_components(graph, directed=False)[1]


def merge(R, U, G, labels, kind):
    """Merged pieces and the exact on-data output error (numerator) of the merge."""
    C = R.shape[1]; dev = R.device
    labels_t = torch.as_tensor(labels, device=dev)
    size = torch.bincount(labels_t, minlength=int(labels_t.max())+1)
    strength = G.diagonal() if kind == 'read' else U.square().sum(0)
    M = torch.zeros(C, C, dtype=G.dtype, device=dev)                 # rows only for merged-away members: e_i = sum_k M_ik (a_k or u_k)
    refs, R_new, U_new = [], [], []
    for group in torch.nonzero(size > 1).flatten().tolist():
        members = torch.nonzero(labels_t == group).flatten()
        ref = members[strength[members].argmax()]
        others = members[members != ref]
        if kind == 'read':
            coef = G[members, ref]/G[ref, ref]                       # a_i ~ coef_i a_ref
            M[others, others] = 1.; M[others, ref] = -coef[members != ref]
            R_new.append(R[:, ref]); U_new.append(U[:, members] @ coef)
        else:
            coef = (U[:, members].T @ U[:, ref])/U[:, ref].square().sum()   # u_i ~ coef_i u_ref
            M[others, others] = 1.; M[others, ref] = -coef[members != ref]
            R_new.append(R[:, members] @ coef); U_new.append(U[:, ref])
        refs.append(members)
    singles = torch.nonzero(size[labels_t] == 1).flatten()
    UU = U.T @ U
    error = float((UU*(M @ G @ M.T)).sum()) if kind == 'read' else float(((M @ UU @ M.T)*G).sum())
    R_out = torch.cat([R[:, singles]]+([torch.stack(R_new, 1)] if R_new else []), 1)
    U_out = torch.cat([U[:, singles]]+([torch.stack(U_new, 1)] if U_new else []), 1)
    return R_out, U_out, error


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='parauto'); ap.add_argument('--random-init', action='store_true')
    ap.add_argument('--sequences', type=int, default=64); ap.add_argument('--score-sequences', type=int, default=16); ap.add_argument('--fisher-k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    from write_attribution import capture_inputs
    from fisher_attribution import capture_fisher
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'))
    model = target()
    if args.method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors'])
        pieces = [(trainer.model.components[n].V.detach().double(), trainer.model.components[n].U.detach().double().T) for n in MODULES]
        del trainer
    else:
        pieces = [(p['read'].cuda().double(), p['write'].cuda().double()) for p in saved['factors']]
    del saved
    weights = [model.get_submodule(n).weight.detach().double() for n in MODULES]
    if args.random_init:
        assert args.method == 'parauto'
        pieces = [(lambda f: (f.read.double(), f.write.double()))(core.initialize_pieces(w.float(), R.shape[1], 0)) for w, (R, _) in zip(weights, pieces)]
        meta['note'] = 'random projected factors at the trained run\'s counts; no training'
    data, manifest = tokens('val')
    started = time.perf_counter()
    # data second moments of each matrix input, held-out tokens
    sigma = [torch.zeros(w.shape[1], w.shape[1], dtype=torch.float64, device='cuda') for w in weights]; n = 0
    for first in range(0, args.sequences, 8):
        rows = torch.tensor(np.array(data[first:first+8], dtype=np.int64), device='cuda')
        with torch.no_grad():
            for s, e in zip(sigma, capture_inputs(model, rows)):
                h = e['h'].double(); s += h.T @ h
        n += rows.shape[0]*(rows.shape[1]-1)
    sigma = [s/n for s in sigma]
    variants = {'before': [(R, U) for R, U in pieces]}
    per_matrix = {}
    for m, (name, (R, U), W, S) in enumerate(zip(MODULES, pieces, weights, sigma)):
        G = R.T @ S @ R
        d = G.diagonal().clamp_min(1e-30).sqrt()
        read_sim = (G/d[:, None]/d[None, :]).abs()
        un = U/U.norm(dim=0).clamp_min(1e-30)
        write_sim = (un.T @ un).abs()
        eye = torch.eye(R.shape[1], dtype=torch.bool, device='cuda')
        read_best = read_sim.masked_fill(eye, 0).max(1).values; write_best = write_sim.masked_fill(eye, 0).max(1).values
        total = float(torch.trace(W @ S @ W.T))
        row = dict(C=R.shape[1], read_partner_max_median=float(read_best.median()), write_partner_max_median=float(write_best.median()))
        for tau in TAUS:
            for kind, sim in (('read', read_sim), ('write', write_sim)):
                labels = components(sim, tau)
                R2, U2, err = merge(R, U, G, labels, kind)
                variants.setdefault(f'{kind}{tau}', []).append((R2, U2))
                row[f'{kind}{tau}'] = dict(partner_share=float(((read_best if kind == 'read' else write_best) > tau).double().mean()),
                                           kept_share=R2.shape[1]/R.shape[1], output_error=err/total)
        per_matrix[name] = row
        del G, read_sim, write_sim
        print(json.dumps({name: {k: (round(v, 4) if isinstance(v, float) else v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()})
                                 for k, v in row.items()}}), flush=True)
    # per-token description length before and after merging, Fisher score units
    counts = {k: [] for k in variants}
    for first in range(0, args.score_sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda')
        cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
        with torch.no_grad():
            for key, ps in variants.items():
                scores = torch.cat([(e['h'].float() @ R.float()).abs()*torch.linalg.vector_norm(e['events']['cot'].float() @ U.float(), dim=1)/np.sqrt(args.fisher_k)
                                    for e, (R, U) in zip(cache, ps)], -1)
                ordered = scores.sort(-1, descending=True).values
                share = ordered.cumsum(-1)/ordered.sum(-1, keepdim=True).clamp_min(1e-30)
                counts[key].append(((share < .8).sum(-1)+1).cpu())
        del cache
        print(json.dumps(dict(score_sequences=first+2, seconds=round(time.perf_counter()-started, 1))), flush=True)
    summary = {}
    for key, ps in variants.items():
        c = torch.cat(counts[key]).double()
        summary[key] = dict(pieces_total=sum(R.shape[1] for R, _ in ps), pieces_for_80_median=float(c.median()), pieces_for_80_mean=float(c.mean()))
        if key != 'before':
            errs = [per_matrix[nm][key]['output_error'] for nm in MODULES]
            summary[key].update(output_error_median=float(np.median(errs)), output_error_max=float(np.max(errs)))
    for key, v in summary.items(): print(key, json.dumps({a: round(b, 4) for a, b in v.items()}), flush=True)
    save_json(dict(method=args.method, random_init=args.random_init, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta,
                   sigma_positions=n, score_sequences=args.score_sequences, gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'],
                   summary=summary, per_matrix=per_matrix, seconds=time.perf_counter()-started), args.output)
    print('REDUNDANCY_PROBE_DONE', flush=True)


if __name__ == '__main__': main()
