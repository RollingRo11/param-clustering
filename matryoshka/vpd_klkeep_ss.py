"""KLKeep for the Goodfire SimpleStories VPD run s-9e774a82 (CI ranking, rows 0-127, keep fractions 0.1-10%), optionally with
some modules always kept (argv[1], comma-separated; fractions then count the rest). Prints {kept: KL}."""
import sys, json; sys.path.insert(0, __import__('os').path.dirname(__file__))
import legacy_ci  # noqa
import numpy as np, torch
from common import tokens, MODULES
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.models.components import make_mask_infos
torch.backends.cuda.matmul.allow_tf32 = False
always = set(sys.argv[1].split(',')) if len(sys.argv) > 1 and sys.argv[1] else set()
ri = ParamDecompRunInfo.from_path('wandb:goodfire/spd/runs/s-9e774a82')
vpd = ComponentModel.from_run_info(ri).cuda().eval().requires_grad_(False)
val, _ = tokens('val')
FR = [.001, .003, .01, .03, .1]
sums = {f: [0., 0, 0] for f in FR}
for first in range(0, 128, 16):
    ids = torch.tensor(np.array(val[first:first+16], dtype=np.int64), device='cuda')[:, :-1]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        base = vpd(ids, cache_type='input')
        ci = vpd.calc_causal_importances(pre_weight_acts=base.cache, detach_inputs=False, sampling='continuous').lower_leaky
        logp = base.output.float().log_softmax(-1)
        scores = [torch.full_like(ci[n].float(), 1e30) if n in always else ci[n].float() for n in MODULES]
        counts = [x.shape[-1] for x in scores]; fixed = sum(c for n, c in zip(MODULES, counts) if n in always)
        order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)
        for f in FR:
            k = fixed + round(f * (sum(counts) - fixed))
            keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
            pred = vpd(ids, mask_infos=make_mask_infos(dict(zip(MODULES, keep.split(counts, -1))))).float()
            sums[f][0] += float((logp.exp() * (logp - pred.log_softmax(-1))).sum()); sums[f][1] += ids.numel(); sums[f][2] = k - fixed
print(json.dumps({str(v[2]): round(v[0] / v[1], 3) for f, v in sums.items()}))
