"""How many components does each method actually consider live per token, and where do they sit?

VPD: counts of causal importance above thresholds, per token, overall and per matrix.
par-auto: the same counts for the normalized Fisher score (share of the token's total score),
plus how many top pieces it takes to hold 50/90/99% of the token's score mass.
This says whether a global top-k at 1-10% is selecting graded importance or breaking ties.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, MODULES, TARGET_SHA, credentials, target, tokens, save_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['parauto', 'vpd'], default='vpd'); ap.add_argument('--sequences', type=int, default=16)
    ap.add_argument('--score', choices=['native', 'fisher'], default='native', help="fisher: score VPD's own pieces with our sensor, so both distributions share units")
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'))
    model = target()
    from train_pair import core
    if args.method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        pieces = [core.FreePieces(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().float().T) for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    del saved
    data, manifest = tokens('val')
    thresholds = [1e-8, 1e-4, .01, .1, .5]
    counts_above = {t: 0. for t in thresholds}; per_matrix = {n: {t: 0. for t in thresholds} for n in MODULES}
    mass = {q: 0. for q in (.5, .9, .99)}; n_tok = 0; total = None
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda'); ids = rows[:, :-1]
        if args.method == 'vpd' and args.score == 'native':
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n].float() for n in MODULES]; del base, ci
        else:
            from fisher_attribution import capture_fisher
            from train_pair import method
            cache = capture_fisher(model, rows, k=8, seed=first)          # needs grad; outside no_grad
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
            del cache
        with torch.no_grad():
            combined = torch.cat(scores, -1)
            total = combined.shape[-1]
            # normalize per token by the largest score, so thresholds mean "relative to this token's top piece"
            scale = combined.amax(-1, keepdim=True).clamp_min(1e-30)
            rel = combined/scale
            for t in thresholds:
                counts_above[t] += float((rel > t).sum())
                for n, s in zip(MODULES, rel.split([x.shape[-1] for x in scores], -1)):
                    per_matrix[n][t] += float((s > t).sum())
            ordered = combined.sort(dim=-1, descending=True).values
            share = ordered.cumsum(-1)/ordered.sum(-1, keepdim=True).clamp_min(1e-30)
            for q in mass: mass[q] += float((share < q).sum(-1).float().sum()+1)
            n_tok += ids.numel(); del combined, rel, ordered, share, scores
    result = dict(method=args.method, score=args.score, checkpoint=str(args.checkpoint), **meta, positions=n_tok, total_components=total,
                  live_per_token={str(t): counts_above[t]/n_tok for t in thresholds},
                  live_fraction={str(t): counts_above[t]/n_tok/total for t in thresholds},
                  pieces_for_score_mass={str(q): mass[q]/n_tok for q in mass},
                  per_matrix={n: {str(t): per_matrix[n][t]/n_tok for t in thresholds} for n in MODULES})
    print(json.dumps({k: result[k] for k in ('live_per_token', 'live_fraction', 'pieces_for_score_mass')}, indent=1), flush=True)
    save_json(result, args.output); print('CI_CENSUS_DONE', flush=True)


if __name__ == '__main__': main()
