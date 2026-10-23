"""Per-piece single ablation (drop c, keep all else) KL per token vs sensor scores (static, per-token Fisher K=8, static
metric from GIM cotangents), per matrix, on static_5k: Spearman within matrix and top-10 overlap, averaged over tokens."""
import sys, json, time; sys.path.insert(0, '.')
import numpy as np, torch
import parauto_core as core, static_fisher, parauto_structured as structured
from common import MODULES, target, tokens
from write_attribution import capture_inputs
from fisher_attribution import capture_fisher
torch.backends.cuda.matmul.allow_tf32 = False
m = target(); val, _ = tokens('val'); train, _ = tokens()
saved = torch.load('/root/runs/matryoshka_ss_mainline/static_5k/checkpoints/005000/model.pth', map_location='cpu', weights_only=False)
pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
counts = [p.read.shape[1] for p in pieces]
metric = [(f['L'].cuda(), f['eigenvalues']) for f in saved['static_metric']]
t0 = time.time(); metric_gim = static_fisher.estimate(m, train, gim=True); print('gim metric s', round(time.time() - t0, 1), flush=True)
rows = torch.from_numpy(np.array(val[0:4], dtype=np.int64)).cuda(); ids = rows[:, :-1]
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    inp = capture_inputs(m, rows)
    static = [structured.scores(e, p).float() for e, p in zip(static_fisher.entries(inp, metric), pieces)]
    gimst = [structured.scores(e, p).float() for e, p in zip(static_fisher.entries(inp, metric_gim), pieces)]
cache = capture_fisher(m, rows, k=8, seed=0, microbatch=4)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    fisher = [structured.scores(e, p).float() for e, p in zip(cache, pieces)]
del cache
# Reference: the all-kept factorized forward; ablate one piece at all positions.
def forward(drop_j=None, drop_c=None):
    hooks = []
    for j, (name, p) in enumerate(zip(MODULES, pieces)):
        def hook(_l, i, _o, p=p, j=j):
            a = i[0] @ p.read
            if j == drop_j: a = a.clone(); a[..., drop_c] = 0
            return a @ p.write.T
        hooks.append(m.get_submodule(name).register_forward_hook(hook))
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return m(ids)[0].float().log_softmax(-1).flatten(0, 1)
    finally:
        for h in hooks: h.remove()
ref = forward(); pref = ref.exp()
t0 = time.time()
true = []
for j, C in enumerate(counts):
    kl = torch.empty(ids.numel(), C, device='cuda')
    for c in range(C):
        out = forward(j, c); kl[:, c] = (pref * (ref - out)).sum(-1)
    true.append(kl)
print('ablations s', round(time.time() - t0, 1), flush=True)
def rank(x): return x.argsort(-1).argsort(-1).float()
def spearman(a, b):
    ra, rb = rank(a), rank(b); ra -= ra.mean(-1, keepdim=True); rb -= rb.mean(-1, keepdim=True)
    return float(((ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)).mean())
def top_overlap(a, b, k=10):
    ta, tb = a.topk(k, -1).indices, b.topk(k, -1).indices
    return float((ta[:, :, None] == tb[:, None, :]).any(-1).float().mean())
res = {}
for j, n in enumerate(MODULES):
    t = true[j]
    res[n] = {name: (round(spearman(s[j], t), 3), round(top_overlap(s[j], t), 3)) for name, s in (('static', static), ('fisher8', fisher), ('gim_static', gimst))}
    res[n]['true_kl_top10_share'] = round(float((t.topk(10, -1).values.sum(-1) / t.sum(-1).clamp_min(1e-12)).mean()), 3)
    print(n, json.dumps(res[n]), flush=True)
json.dump(res, open('/root/runs/ablation_vs_score_static5k.json', 'w'), indent=1)
