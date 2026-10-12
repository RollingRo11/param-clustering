"""CPU checks for learned ownership: identity ownership = rank-one scores, soft one-hot = hard, short exact-sum fit."""
import math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'par-auto'))
import parauto_core as core, parauto_method as method, parauto_structured as structured
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
n, K, din, dout, N = 64, 3, 7, 6, 16
pieces = core.FreePieces(torch.randn(din, N), torch.randn(dout, N))
h = torch.randn(n, din)
entries = dict(scalar_loss=dict(structured=True, kind='scalar_loss', h=h, events={'g': torch.randn(n, dout)}, shared={}, net_rms=torch.ones(n)),
               vjp=dict(structured=True, kind='vjp', h=h, events={'cot': torch.randn(n, K, dout)}, output_count=K, shared={}, net_rms=torch.ones(n)))
owner = torch.randint(0, 4, (N,)); onehot = torch.nn.functional.one_hot(owner, 4).double()
for kind, e in entries.items():
    torch.testing.assert_close(structured.soft_component_scores(e, pieces, torch.eye(N)), structured.scores(e, pieces))
    torch.testing.assert_close(structured.soft_component_scores(e, pieces, onehot), structured.hard_component_scores(e, pieces, owner, 4))
    for m in range(4):                                             # brute force: whole owned sub-matrix
        P = pieces.write[:, owner == m] @ pieces.read[:, owner == m].T
        if kind == 'scalar_loss': ref = torch.einsum('nd,dm,nm->n', e['events']['g'], P, h).abs()
        else: ref = torch.einsum('nqd,dm,nm->nq', e['events']['cot'], P, h).norm(dim=1)/math.sqrt(K)
        torch.testing.assert_close(structured.hard_component_scores(e, pieces, owner, 4)[:, m], ref)
    print(kind, 'identity = rank-one, one-hot soft = hard = brute force')
weight = torch.randn(dout, din); initial = core.initialize_pieces(weight, N, 0)
cfg = method.TrainConfig(steps=40, batch=n, log_interval=10, ownership_ratio=4, rank_price=0.2, lr=1e-2, lr_min=1e-3)
logits0 = method.initial_ownership(N, cfg, 'cpu')
torch.testing.assert_close(method.ownership_matrix(logits0).max(-1).values.double(), torch.full((N,), 0.5), rtol=1e-5, atol=1e-6)
assert (torch.bincount(logits0.argmax(-1), minlength=4) == 4).all()
states, rows = [], []
fitted, _ = method.fit([entries['scalar_loss']], [weight], [initial], cfg, log=rows.append, checkpoint=states.append, checkpoint_interval=40)
assert core.reconstruction_error(weight, fitted[0]) < 1e-6
assert all(math.isfinite(r['loss']) and 'components_used' in r for r in rows), rows[-1]
moved = float((states[-1]['ownership'][0] - logits0).abs().max())
assert moved > 1e-3, moved
print('fit ok', {k: rows[-1][k] for k in ('loss', 'components_used', 'component_size_max', 'ownership_max_mean', 'reconstruction_relative_error')}, 'ownership moved', round(moved, 4))
states2 = []
cfg_resume = method.TrainConfig(steps=60, batch=n, log_interval=10, ownership_ratio=4, rank_price=0.2, lr=1e-2, lr_min=1e-3)
resume = {**states[-1], 'config': method.asdict(cfg_resume)}
method.fit([entries['scalar_loss']], [weight], [initial], cfg_resume, log=lambda r: None, resume=resume, checkpoint=states2.append, checkpoint_interval=20)
assert states2[-1]['next_step'] == 60
print('resume ok')
cfg_h = method.TrainConfig(steps=40, batch=n, log_interval=10, ownership_ratio=4, rank_price=0.2, lr=1e-2, lr_min=1e-3, ownership_hard=True)
logits = (method.initial_ownership(N, cfg_h, 'cpu') + 0.1*torch.randn(N, 4)).requires_grad_()
st = method.ownership_forward(logits, cfg_h)
hard_owner = logits.argmax(-1)
for kind, e in entries.items():
    torch.testing.assert_close(structured.soft_component_scores(e, pieces, st.detach()), structured.hard_component_scores(e, pieces, hard_owner, 4), rtol=1e-5, atol=1e-6)   # shares are float32
cost = method.component_cost(structured.soft_component_scores(entries['vjp'], pieces, st), st.sum(0), cfg_h, 0.7).mean()
torch.testing.assert_close(st.detach().sum(0).double(), torch.bincount(hard_owner, minlength=4).double(), rtol=1e-5, atol=1e-5)
cost.backward(); assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
print('straight-through forward = hard ownership (scores and counts), gradient reaches logits')
states_h, rows_h = [], []
fitted_h, _ = method.fit([entries['scalar_loss']], [weight], [initial], cfg_h, log=rows_h.append, checkpoint=states_h.append, checkpoint_interval=40)
assert core.reconstruction_error(weight, fitted_h[0]) < 1e-6 and all(math.isfinite(r['loss']) for r in rows_h)
print('straight-through fit ok', {k: rows_h[-1][k] for k in ('loss', 'components_used', 'component_size_max', 'ownership_max_mean')},
      'ownership moved', round(float((states_h[-1]['ownership'][0] - method.initial_ownership(N, cfg_h, 'cpu')).abs().max()), 4))
print('OWNERSHIP_CHECK_OK')
