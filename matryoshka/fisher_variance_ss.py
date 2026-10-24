"""Per-token score = read factor |r.h| x write factor sqrt(u^T F(x) u). How much of the log score's variation across tokens
(for each piece) is the write factor? Write factor from two independent K=8 Fisher draws (split-half) to separate token
variation from sampling noise. static_5k and B at step 5000, 2,048 val tokens."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np, torch
from common import tokens, MODULES, target
from fisher_attribution import capture_fisher
torch.backends.cuda.matmul.allow_tf32 = False
m = target(); val, _ = tokens('val')
rows = torch.from_numpy(np.array(val[:4], dtype=np.int64)).cuda()
c1 = capture_fisher(m, rows, k=8, seed=1, microbatch=4); c2 = capture_fisher(m, rows, k=8, seed=2, microbatch=4)
import json
out = {}
for run in ('fs30_adv50_5k',):
    saved = torch.load(f'/root/runs/matryoshka_ss_mainline/{run}/checkpoints/005000/model.pth', map_location='cpu', weights_only=False)
    tot = {'read': [], 'write': [], 'noise': [], 'share_write': []}
    for n, e1, e2, f in zip(MODULES, c1, c2, saved['factors']):
        R, U = f['read'].cuda().float(), f['write'].cuda().float()
        a = (e1['h'].float() @ R).abs().clamp_min(1e-12).log()                      # [T, C] log read factor
        w1 = ((e1['events']['cot'].float() @ U) ** 2).mean(1).clamp_min(1e-24).log() / 2   # log write factor, draw 1
        w2 = ((e2['events']['cot'].float() @ U) ** 2).mean(1).clamp_min(1e-24).log() / 2   # draw 2
        # weight pieces by how often they matter: only (token, piece) pairs in the token's top 10% by score
        s = a + (w1 + w2) / 2
        keep = s >= s.quantile(0.9, dim=-1, keepdim=True)
        def var(x):  # variance across tokens per piece, averaged over pieces, on the kept pairs
            xm = torch.where(keep, x, torch.nan); mu = xm.nanmean(0, keepdim=True)
            return float(((xm - mu) ** 2).nanmean(0).nanmean())
        vr, vw, vn = var(a), var((w1 + w2) / 2), var((w1 - w2) / 2)       # noise: half-difference of two draws
        wm = (w1 + w2) / 2; resid = wm - wm.mean(-1, keepdim=True)            # remove each token's shared level for this matrix
        dm = (w1 - w2) / 2; dres = dm - dm.mean(-1, keepdim=True)
        vres, vnres = var(resid), var(dres)
        tot.setdefault('resid', []).append(max(vres - vnres / 2, 0))
        out[n] = dict(read=vr, write=max(vw - vn / 2, 0), write_piece=max(vres - vnres / 2, 0), noise=vn / 2)
        tot['read'].append(vr); tot['write'].append(max(vw - vn / 2, 0)); tot['noise'].append(vn / 2)
        print(f'{run:14s} {n:18s} read {vr:6.3f}  write {max(vw - vn / 2, 0):6.3f}  write minus token-shared level {max(vres - vnres / 2, 0):6.3f}')
    print(run, 'mean over matrices: read', round(np.mean(tot['read']), 3), 'write (token part)', round(np.mean(tot['write']), 3), 'noise', round(np.mean(tot['noise']), 3), 'write piece-specific', round(np.mean(tot['resid']), 3))
json.dump(out, open('/root/runs/fisher_variance_B.json', 'w'), indent=1)
