"""CPU checks for rank-k pieces: rank-1 equivalence, brute-force scores, singular values, a short exact-sum fit."""
import math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'par-auto'))
import parauto_core as core, parauto_method as method, parauto_structured as structured
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
n, K, din, dout, k, C = 64, 3, 7, 6, 4, 5
cols = C*k
entry = dict(structured=True, kind='vjp', h=torch.randn(n, din), events={'cot': torch.randn(n, K, dout)}, output_count=K,
             net_rms=torch.ones(n), shared={})
pieces = core.FreePieces(torch.randn(din, cols), torch.randn(dout, cols))
torch.testing.assert_close(structured.grouped_scores(entry, pieces, 1), structured.scores(entry, pieces)); print('rank-1 score equals rank-one score')
g = structured.grouped_scores(entry, pieces, k)
for c in range(C):
    P = pieces.write[:, c*k:(c+1)*k] @ pieces.read[:, c*k:(c+1)*k].T
    brute = torch.einsum('nqd,dm,nm->nq', entry['events']['cot'], P, entry['h']).norm(dim=1)/math.sqrt(K)
    torch.testing.assert_close(g[:, c], brute)
    torch.testing.assert_close(core.piece_singular_values(pieces, k)[c], torch.linalg.svdvals(P)[:k])
print('rank-k scores and singular values match brute force')
scalar = dict(structured=True, kind='scalar_loss', h=entry['h'], events={'g': torch.randn(n, dout)}, shared={}, net_rms=torch.ones(n))
torch.testing.assert_close(structured.grouped_scores(scalar, pieces, 1), structured.scores(scalar, pieces))
gs = structured.grouped_scores(scalar, pieces, k)
for c in range(C):
    P = pieces.write[:, c*k:(c+1)*k] @ pieces.read[:, c*k:(c+1)*k].T
    torch.testing.assert_close(gs[:, c], torch.einsum('nd,dm,nm->n', scalar['events']['g'], P, entry['h']).abs())
print('scalar_loss (GIM) whole-piece scores match rank-one and brute force')
cfg_s = method.TrainConfig(steps=20, batch=n, log_interval=10, piece_rank=k, rank_coeff=1.0, lr=1e-2, lr_min=1e-3, precision='bf16', projection_precision='fp32')
if torch.cuda.is_available():
    _, _ = method.fit([{**scalar, 'h': scalar['h'].cuda().float(), 'events': {'g': scalar['events']['g'].cuda().float()}, 'net_rms': torch.ones(n, device='cuda')}],
                      [torch.randn(dout, din, device='cuda', dtype=torch.float32)], [core.initialize_pieces(torch.randn(dout, din, device='cuda', dtype=torch.float32), cols, 0)], cfg_s, log=lambda r: None)
    print('bf16 scalar_loss fit ok on GPU')
weight_s = torch.randn(dout, din); initial_s = core.initialize_pieces(weight_s, cols, 0)
fitted_s, _ = method.fit([scalar], [weight_s], [initial_s], method.TrainConfig(steps=20, batch=n, log_interval=10, piece_rank=k, rank_coeff=1.0, lr=1e-2, lr_min=1e-3), log=lambda r: None)
assert core.reconstruction_error(weight_s, fitted_s[0]) < 1e-6
print('scalar_loss fit ok')
weight = torch.randn(dout, din)
initial = core.initialize_pieces(weight, cols, 0)
rows = []
cfg = method.TrainConfig(steps=40, batch=n, log_interval=10, piece_rank=k, rank_coeff=1.0, lr=1e-2, lr_min=1e-3)
fitted, hist = method.fit([entry], [weight], [initial], cfg, log=rows.append)
assert core.reconstruction_error(weight, fitted[0]) < 1e-6
assert all(math.isfinite(r['loss']) and 'effective_rank_mean' in r and 'rank_normalized' in r for r in rows), rows[-1]
print('fit ok', {k_: rows[-1][k_] for k_ in ('loss', 'effective_rank_mean', 'participation_rank_mean', 'rank_normalized', 'reconstruction_relative_error')})
assert {**method.asdict(method.TrainConfig()), **{k_: v for k_, v in method.asdict(cfg).items() if k_ not in ('piece_rank', 'rank_coeff')}} != method.asdict(cfg)
print('RANK_PIECES_CHECK_OK')
