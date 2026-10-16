"""Per-piece weight mass for the matryoshka67 comparison checkpoints.

mass_c = ||u_c r_c^T||_F^2 = ||u_c||^2 ||r_c||^2, as a share of the model's weight mass sum_m ||W_m||_F^2 over the
24 decomposed matrices (the fro_share of the earlier cofac/VPD mass plots). If pieces were orthogonal the shares would
sum to 100%; a larger total means pieces are big and cancel. VPD's delta (W - sum of pieces) is reported separately.
Usage: weight_mass67.py OUTPUT_JSON [LABEL ...]   (default: every checkpoint below)
"""
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch

from common import MODULES, TARGET_SHA, credentials, target, save_json, sha

EXP = ROOT/'output/matryoshka67_20260923'
CHECKPOINTS = {
    'matryoshka_hard_5000': ('parauto', EXP/'main_hard/checkpoints/005000/model.pth'),
    'matryoshka_soft_5000': ('parauto', EXP/'main/checkpoints/005000/model.pth'),
    'fisher_k4_5000': ('parauto', ROOT/'output/fisher_k4_5000/checkpoints/005000/model.pth'),
    'vpd_5000': ('vpd', ROOT/'checkpoints/step_5000/vpd.pth'),
    'vpd_published_400000': ('vpd', ROOT/'checkpoints/published_vs_finished/vpd.pth'),
    'matryoshka_lp1_5000': ('parauto', EXP/'main_hard_lp1/checkpoints/005000/model.pth'),
    'matryoshka_lp1_newbudgets_5000': ('parauto', EXP/'main_hard_lp1_strat_per1_pow1.5/checkpoints/005000/model.pth'),
    'matryoshka_lp1_newbudgets_adv1_5000': ('parauto', EXP/'main_hard_lp1_strat_per1_pow1.5_adv1/checkpoints/005000/model.pth'),
}


def pieces_of(kind, path, model):
    """[(read_norm2 [C], write_norm2 [C], reconstruction [d_out, d_in])] per matrix, float64."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['target_sha256'] == TARGET_SHA
    out = []
    if kind == 'parauto':
        for p in saved['factors']:
            r, u = p['read'].double(), p['write'].double()                  # [d_in, C], [d_out, C]
            out.append((r.pow(2).sum(0), u.pow(2).sum(0), u @ r.T))
        return out, saved.get('completed')
    # Read VPD's U and V straight from the state dict (CPU only; no component model on a GPU).
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    state = saved['factors']
    handle_deprecated_state_dict_keys_(state)
    for name in MODULES:
        found = {}
        for part in ('U', 'V'):
            keys = [k for k in state if k.endswith('.' + part) and not k.startswith('target_model.')
                    and (('.' + name + '.') in ('.' + k) or ('.' + name.replace('.', '-') + '.') in ('.' + k))]
            assert len(keys) == 1, (name, part, keys[:5])
            found[part] = state[keys[0]].double()
        V, U = found['V'], found['U']                                            # expected [d_in, C], [C, d_out]
        W = model.get_submodule(name).weight
        assert V.shape[0] == W.shape[1] and U.shape[1] == W.shape[0] and V.shape[1] == U.shape[0], (name, V.shape, U.shape, W.shape)
        out.append((V.pow(2).sum(0), U.pow(2).sum(1), (V @ U).T))
    return out, saved.get('completed')


def main():
    output, labels = Path(sys.argv[1]), sys.argv[2:] or list(CHECKPOINTS)
    credentials()
    model = target('cpu')
    weights = [model.get_submodule(n).weight.detach().double().cpu() for n in MODULES]
    total = float(sum(w.pow(2).sum() for w in weights))
    result = dict(model_weight_mass=total, definition=__doc__.split('Usage')[0].strip(), methods={})
    for label in labels:
        kind, path = CHECKPOINTS[label]
        parts, step = pieces_of(kind, path, model)
        mass = torch.cat([rn * wn for rn, wn, _ in parts]) / total
        residual = sum(float((w - rec).pow(2).sum()) for w, (_, _, rec) in zip(weights, parts)) / total
        shares = np.sort(mass.numpy())[::-1]
        cum = np.cumsum(shares) / shares.sum()
        result['methods'][label] = dict(
            kind=kind, step=step, checkpoint=str(path), checkpoint_sha256=sha(path), pieces=int(len(shares)),
            total_share=float(shares.sum()), largest_share=float(shares[0]),
            pieces_for_50pct=int((cum < .5).sum()) + 1, pieces_for_90pct=int((cum < .9).sum()) + 1,
            residual_share=residual,                    # VPD: its delta; par-auto: exact-sum projection error
            sorted_shares=[float('%.5g' % v) for v in shares])
        print(label, {k: v for k, v in result['methods'][label].items() if k != 'sorted_shares'}, flush=True)
        del parts, mass
    save_json(result, output)


if __name__ == '__main__': main()
