"""Top-activating held-out contexts per component, for reading and intruder tests.

Activation = the score each method ranks by in KLKeep (par-auto: its sensor; VPD: CI).
Each sequence contributes at most one example per component (its max position), so a
component cannot look coherent by repeating one document. Val rows 128.. are used;
KLKeep uses rows 0..127. A component "fires" on a token when it is in that token's
top 1% across all 24 matrices (the KLKeep 1% keep set).

Windows are saved as token ids (40 before, 8 after); decoding happens locally.
For residual writers (o_proj, down_proj) the top tokens of lm_head(ln_f.weight * u)
are also saved: what the write pushes up or down, ignoring later layers.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json
from train_pair import VPD, capture, core, method

BEFORE, AFTER = 40, 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['parauto', 'vpd'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sensor', choices=['gradient', 'gim', 'write'], default='gim')
    parser.add_argument('--first-row', type=int, default=128)
    parser.add_argument('--sequences', type=int, default=1920)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--sample', type=int, default=800)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); credentials(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    model = target()
    if args.method == 'vpd':
        trainer = VPD(model, evaluation=True)
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        writes = [trainer.model.components[n].U.T.float() if hasattr(trainer.model.components[n], 'U') else None for n in MODULES]
    else:
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
        writes = [p.write for p in pieces]
    del saved
    data, _ = tokens('val')
    rows_end = min(len(data), args.first_row+args.sequences)
    counts, vals, seqs, poss, fire, n_tok = None, None, None, None, None, 0
    started = time.perf_counter()
    for first in range(args.first_row, rows_end, args.batch):
        rows = torch.tensor(np.array(data[first:min(first+args.batch, rows_end)], dtype=np.int64), device='cuda')
        ids = rows[:, :-1]
        if args.method == 'vpd':
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
                scores = [ci.lower_leaky[n].float() for n in MODULES]
        else:
            if args.sensor == 'gim':
                from gim_attribution import capture_gim
                cache = capture_gim(model, rows)
            elif args.sensor == 'write':
                from write_attribution import capture_inputs
                cache = capture_inputs(model, rows)
            else:
                cache = capture(model, rows)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                scores = [method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)]
            del cache
        with torch.no_grad():
            counts = counts or [s.shape[-1] for s in scores]
            combined = torch.cat(scores, -1)                                   # [B,T,C_total]
            del scores
            k = max(1, round(.01*combined.shape[-1]))
            threshold = combined.topk(k, dim=-1).values[..., -1:]
            hits = (combined >= threshold).sum((0, 1))
            fire = hits if fire is None else fire+hits
            n_tok += ids.numel()
            best, where = combined.max(1)                                      # one example per sequence
            seq = torch.arange(first, first+len(rows), device='cuda')[:, None].expand_as(best)
            if vals is None:
                vals, seqs, poss = best, seq, where
            else:
                vals, seqs, poss = torch.cat((vals, best)), torch.cat((seqs, seq)), torch.cat((poss, where))
            if len(vals) > args.topk:
                top = vals.topk(args.topk, dim=0).indices
                vals, seqs, poss = vals.gather(0, top), seqs.gather(0, top), poss.gather(0, top)
            del combined
        if (first-args.first_row) % (args.batch*40) == 0:
            print(json.dumps(dict(rows_done=first+len(rows)-args.first_row, seconds=time.perf_counter()-started)), flush=True)
    names = [n for n, c in zip(MODULES, counts) for _ in range(c)]
    local = [i for c in counts for i in range(c)]
    fire = fire.cpu()
    alive = torch.nonzero(fire > 0).flatten()
    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(alive.numpy(), size=min(args.sample, len(alive)), replace=False).tolist())
    vals, seqs, poss = vals.cpu(), seqs.cpu(), poss.cpu()
    unembed = model.lm_head.weight.float() * model.ln_f.weight.float()               # [V,768]
    offsets = np.cumsum([0]+counts)
    components = {}
    for c in sample:
        examples = []
        for j in range(vals.shape[0]):
            row, pos = int(seqs[j, c]), int(poss[j, c])
            lo = max(0, pos-BEFORE)
            window = np.array(data[row, lo:pos+1+AFTER], dtype=np.int64).tolist()
            examples.append(dict(score=float(vals[j, c]), row=row, pos=pos, ids=window, mark=pos-lo))
        entry = dict(matrix=names[c], index=local[c], fire_rate=float(fire[c])/n_tok, examples=examples)
        m = MODULES.index(names[c])
        if names[c].endswith(('o_proj', 'down_proj')) and writes[m] is not None:
            u = writes[m][:, local[c]].float().cuda()
            logits = unembed @ (u/u.norm())
            entry['promotes'] = logits.topk(10).indices.tolist()
            entry['suppresses'] = (-logits).topk(10).indices.tolist()
        components[str(c)] = entry
    per_matrix = {n: dict(components=c, alive=int((fire[o:o+c] > 0).sum()),
                          median_fire_rate=float(fire[o:o+c].float().median())/n_tok)
                  for n, c, o in zip(MODULES, counts, offsets)}
    save_json(dict(method=args.method, sensor=args.sensor if args.method == 'parauto' else 'VPD CI',
                   checkpoint=str(args.checkpoint), rows=[args.first_row, rows_end], tokens=n_tok,
                   total_components=int(sum(counts)), alive=int(len(alive)), per_matrix=per_matrix,
                   fire_rate_quantiles={q: float(torch.quantile(fire.float()/n_tok, q)) for q in (.1, .5, .9, .99)},
                   seconds=time.perf_counter()-started, components=components), args.output)
    print('INTERP_EVIDENCE_DONE', flush=True)


if __name__ == '__main__': main()
