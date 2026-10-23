"""Per-matrix diagnostics of a SimpleStories par-auto checkpoint (static sensor; the run's own floors if it has them),
on val rows 0-31: mean kept pieces per matrix in the global top-k (default 71); for floored runs, the pieces above their
floor per token (the analogue of VPD's CI > 0 count); the KL with only one MLP matrix masked to VPD's mean count there
(others at full weight); and the overall KL at k. Usage: split_check_ss.py CHECKPOINT [K]"""
import json
import os
import sys

import numpy as np
import torch

import parauto_core as core
import parauto_structured as structured
import static_fisher
from common import MODULES, target, tokens
from write_attribution import capture_inputs

VPD_COUNT = dict(zip(MODULES, [1.0, 1.2, 0.8, 1.5, 5.4, 5.1, 1.9, 2.0, 2.7, 6.3, 9.2, 10.1]))   # s-9e774a82, CI > 0


def main():
    path, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 71
    torch.backends.cuda.matmul.allow_tf32 = False
    m = target(); val, _ = tokens('val')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    metric = ([(f['L'].cuda(), f['eigenvalues']) for f in saved['static_metric']] if 'static_metric' in saved else
              static_fisher.cached(m, os.environ.get('MATRYOSHKA_STATIC_CACHE', '/root/data/ss/static_fisher.pt')))
    floor = saved['score_floor'].cuda() if 'score_floor' in saved else None
    counts = [p.read.shape[1] for p in pieces]
    owner = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).cuda()
    kept, above, single, overall, n = torch.zeros(12), torch.zeros(12), {}, 0., 0
    for first in range(0, 32, 8):
        rows = torch.from_numpy(np.array(val[first:first+8], dtype=np.int64)).cuda(); ids = rows[:, :-1]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            raw = [structured.scores(e, p).float() for e, p in zip(static_fisher.entries(capture_inputs(m, rows), metric), pieces)]
            if floor is not None:
                parts = []
                for x, f in zip(raw, floor.split(counts)):
                    parts.append(torch.nn.functional.softplus((x - f) / (f / 10)) * (f / 10) + 1e-6 * x)
                    above[len(parts) - 1] += float((x > f).float().sum(-1).sum())
                raw = parts
            score = torch.cat(raw, -1)
            logp = m(ids)[0].float().log_softmax(-1)

            def kl(masks):
                hooks = [m.get_submodule(name).register_forward_hook(
                    lambda _l, i, _o, p=p, k=k: ((i[0] @ p.read) * k.view(*i[0].shape[:-1], -1)) @ p.write.T)
                    for name, p, k in zip(MODULES, pieces, masks) if k is not None]
                try:
                    out = m(ids)[0].float().log_softmax(-1)
                finally:
                    for h in hooks: h.remove()
                return float((logp.exp() * (logp - out)).sum(-1).mean())

            top = score.topk(K, -1).indices
            kept += torch.stack([(owner[top] == j).float().sum(-1).sum() for j in range(12)]).cpu()
            overall += kl(list(torch.zeros_like(score).scatter_(-1, top, 1.).split(counts, -1)))
            for j, name in enumerate(MODULES):
                if 'mlp' not in name: continue
                k = max(1, round(VPD_COUNT[name]))
                mask = torch.zeros_like(raw[j]).scatter_(-1, raw[j].topk(k, -1).indices, 1.)
                single[name] = single.get(name, 0.) + kl([mask if i == j else None for i in range(12)])
        n += ids.numel() if False else 1
    tokens_n = 32 * 512
    print(json.dumps(dict(checkpoint=path, k=K, kl_at_k=round(overall / n, 3),
                          kept_per_matrix={nm: round(float(v) / tokens_n, 1) for nm, v in zip(MODULES, kept)},
                          above_floor_per_matrix=({nm: round(float(v) / tokens_n, 1) for nm, v in zip(MODULES, above)} if floor is not None else None),
                          single_mlp_kl_at_vpd_count={nm: round(v / n, 3) for nm, v in single.items()}), indent=1))


if __name__ == '__main__':
    main()
