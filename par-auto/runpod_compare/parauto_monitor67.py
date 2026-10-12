"""A par-auto checkpoint on matryoshka67's held-out monitor (same 4 contexts, budgets and Fisher K=4 ranking as
the training-time monitor).  Usage: parauto_monitor67.py CHECKPOINT OUTPUT_JSON"""
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch

import matryoshka67 as mt
from common import TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core


def main():
    path, output = Path(sys.argv[1]), Path(sys.argv[2])
    credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['method'] == 'parauto' and saved['target_sha256'] == TARGET_SHA
    pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    assert [p.count for p in pieces] == mt.COUNTS
    model = target()
    val, manifest = tokens('val')
    rows = torch.from_numpy(np.array(val[mt.MONITOR_ROWS], dtype=np.int64)).cuda()
    curve = mt.monitor(model, rows, pieces, seed=77)
    result = dict(step=saved['completed'], monitor=curve, total_components=sum(mt.COUNTS),
                  rows=[mt.MONITOR_ROWS.start, mt.MONITOR_ROWS.stop], ranking=f'Fisher K={mt.FISHER_K}, seed 77, hard global top-k',
                  checkpoint=str(path), checkpoint_sha256=sha(path), val_sha256=manifest['val']['sha256'])
    save_json(result, output)
    print(result, flush=True)


if __name__ == '__main__': main()
