"""What kept owned components contain: per keep fraction, components kept per token (L0) and the rank-one pieces they own.

Same ranking as KLkeep (Fisher K=8, hard argmax ownership, global top-k per token over all matrices). Also reports each
matrix's largest components as a share of that matrix's pieces, so a low L0 made of near-whole matrices is visible.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method
from fisher_attribution import capture_fisher

FRACTIONS = [.0003, .001, .003, .01, .03, .1, .2, .4]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sequences', type=int, default=16); ap.add_argument('--fisher-k', type=int, default=8)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'parauto' and saved['target_sha256'] == TARGET_SHA and 'ownership_logits' in saved['factors'][0]
    pieces, owners, sizes, per_matrix = [], [], [], {}
    for name, f in zip(MODULES, saved['factors']):
        pieces.append(core.FreePieces(f['read'].cuda(), f['write'].cuda()))
        used, compact = torch.unique(f['ownership_logits'].argmax(-1), return_inverse=True)
        size = torch.bincount(compact, minlength=len(used))
        owners.append((compact.cuda(), len(used))); sizes.append(size.cuda())
        top = size.sort(descending=True).values
        per_matrix[name] = dict(pieces=int(size.sum()), components=len(used), largest=top[:5].tolist(),
                                largest_share=float(top[0]/size.sum()), share_in_components_over_64=float(size[size > 64].sum()/size.sum()))
    completed = saved.get('completed'); del saved
    model = target(); data, manifest = tokens('val')
    all_sizes = torch.cat(sizes).float(); total_pieces = float(all_sizes.sum())
    acc = {f: dict(l0=[], pieces=[], pieces_in_big=[]) for f in FRACTIONS}
    started = time.perf_counter()
    for first in range(0, args.sequences, 2):
        rows = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device='cuda')
        cache = capture_fisher(model, rows, k=args.fisher_k, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            scores = torch.cat([method.structured.hard_component_scores(e, p, o, m).float() for e, p, (o, m) in zip(cache, pieces, owners)], -1)
        del cache
        with torch.no_grad():
            order = scores.argsort(dim=-1, descending=True, stable=True)
            for f in FRACTIONS:
                k = max(1, round(f*scores.shape[-1]))
                kept = all_sizes[order[:, :k]]                                   # [tokens, k] pieces owned by each kept component
                acc[f]['l0'].append(k); acc[f]['pieces'].append(kept.sum(-1).cpu())
                acc[f]['pieces_in_big'].append((kept*(kept > 64)).sum(-1).cpu())
        print(json.dumps(dict(sequences=first+2, seconds=round(time.perf_counter()-started, 1))), flush=True)
    summary = {}
    for f, d in acc.items():
        p = torch.cat(d['pieces']); big = torch.cat(d['pieces_in_big'])
        summary[str(f)] = dict(components_kept=d['l0'][0], pieces_kept_median=float(p.median()), pieces_kept_mean=float(p.mean()),
                               pieces_kept_share_of_all=float(p.mean()/total_pieces), share_of_kept_pieces_in_components_over_64=float(big.sum()/p.sum().clamp_min(1)))
        print(f, json.dumps({a: round(b, 4) if isinstance(b, float) else b for a, b in summary[str(f)].items()}), flush=True)
    for name, v in per_matrix.items(): print(name, json.dumps(v), flush=True)
    save_json(dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), completed=completed, total_components=int(all_sizes.numel()),
                   total_pieces=int(total_pieces), sequences=args.sequences, gpu=torch.cuda.get_device_name(), data_sha256=manifest['val']['sha256'],
                   by_keep_fraction=summary, per_matrix=per_matrix, seconds=time.perf_counter()-started), args.output)
    print('OWNED_L0_DONE', flush=True)


if __name__ == '__main__': main()
