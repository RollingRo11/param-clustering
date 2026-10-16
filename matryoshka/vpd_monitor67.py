"""A VPD checkpoint on matryoshka67's held-out monitor: the same 4 contexts (val rows 1000-1003), the same
absolute per-token budgets, hard global top-k by VPD's native CI (lower_leaky, continuous), delta excluded,
exactly as evaluate_checkpoint.py masks VPD.  Usage: vpd_monitor67.py CHECKPOINT OUTPUT_JSON"""
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch

import matryoshka67 as mt
from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import VPD


def main():
    path, output = Path(sys.argv[1]), Path(sys.argv[2])
    credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['method'] == 'vpd' and saved['target_sha256'] == TARGET_SHA
    model = target()
    trainer = VPD(model, evaluation=True)
    from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
    from param_decomp.models.components import make_mask_infos
    handle_deprecated_state_dict_keys_(saved['factors'])
    trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
    val, manifest = tokens('val')
    ids = torch.from_numpy(np.array(val[mt.MONITOR_ROWS], dtype=np.int64)).cuda()[:, :-1]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        base = trainer.model(ids, cache_type='input')
        ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
        scores = [ci.lower_leaky[n] for n in MODULES]
        logp = base.output.float().log_softmax(-1)
        counts = [s.shape[-1] for s in scores]
        combined = torch.cat(scores, -1).float()
        order = combined.argsort(dim=-1, descending=True, stable=True)
        positive = float((combined > 0).sum(-1).float().mean())
        curve = {}
        for k in mt.MONITOR_BUDGETS + (sum(counts),):
            keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
            pred = trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, keep.split(counts, -1)))))
            curve[k] = float(mt.kl_per_position(logp, pred).mean())
    result = dict(step=saved['completed'], monitor=curve, total_components=sum(counts), positive_ci_per_token=positive,
                  rows=[mt.MONITOR_ROWS.start, mt.MONITOR_ROWS.stop], ranking='VPD CI lower_leaky, continuous; delta excluded',
                  checkpoint=str(path), checkpoint_sha256=sha(path), val_sha256=manifest['val']['sha256'])
    save_json(result, output)
    print(result, flush=True)


if __name__ == '__main__': main()
