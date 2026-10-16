"""Per-token VPD causal importance of every piece in the CI > 0.1 co-activation groups (from coactive_vpd67.py), so group
programs can be written and checked on a laptop. Saves [tokens, pieces] CI (fp16) for the group pieces, val rows 0..rows-1.
Usage (from code/vpd): dump_group_ci67.py --checkpoint vpd.pth --groups vpd_published.json --rows 256 --device cpu --output OUT.pt"""
import argparse
import contextlib
import json

import numpy as np
import torch

from common import MODULES, TARGET_SHA, credentials, target, tokens

ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint'); ap.add_argument('--groups'); ap.add_argument('--output')
ap.add_argument('--rows', type=int, default=256); ap.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
a = ap.parse_args(); credentials(); dev = a.device
if dev == 'cpu': torch.nn.Module.cuda = lambda self, *x, **k: self      # train_pair.VPD calls .cuda()
precision = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if dev == 'cuda' else contextlib.nullcontext
from param_decomp.utils.distributed_utils import init_distributed
init_distributed()
from train_pair import VPD
from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False, mmap=True)
assert saved['method'] == 'vpd' and saved['target_sha256'] == TARGET_SHA
trainer = VPD(target(dev), evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False); del saved
groups = [dict(matrix=m, pieces=g['pieces']) for m, v in json.load(open(a.groups))['thresholds']['0.1']['matrices'].items()
          for g in v['groups']]
data, _ = tokens('val'); out = []
for first in range(0, a.rows, 2):
    ids = torch.tensor(np.array(data[first:first+2], dtype=np.int64), device=dev)[:, :-1]
    with torch.no_grad(), precision():
        base = trainer.model(ids, cache_type='input')
        ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous')
    out.append(torch.cat([ci.lower_leaky[g['matrix']][..., g['pieces']].float().flatten(0, 1) for g in groups], -1).half().cpu())
    if first % 32 == 0: print(dict(rows=first + 2), flush=True)
torch.save(dict(groups=groups, ci=torch.cat(out), rows=a.rows, device=dev), a.output); print('DUMP_DONE', flush=True)
