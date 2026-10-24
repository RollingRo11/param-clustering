"""(Copied from the 20 Oct SS port, pgd/vpd_ss_eval.py; fractions now EVAL_FRACTIONS.) Usage: vpd_eval_ss.py wandb:goodfire/spd/runs/s-9e774a82 OUT_DIR
KLKeep and the global adversarial probe for a Goodfire SimpleStories VPD decomposition, on the same 128 data_ss
validation rows (65,536 positions) the matryoshka runs are evaluated on.

KLKeep: as compare/evaluate_checkpoint.py (vpd): CI ranks pieces globally per position across all 12 matrices,
hard keep masks at the same keep fractions, delta excluded, KL and CE summed in fp32.
Probe: as compare/adversarial_probe.py --scope global (vpd): m = g + (1 - g) r, g native CI or a 0/1 top-k keep set,
one r per piece for all 128 sequences, random init, sign steps of 0.1 clamped to [0, 1], KL at 0..320 steps."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
os.environ.setdefault('MATRYOSHKA_TARGET', 'simplestories')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import legacy_ci  # noqa: Feb-2026 CI network input path
from common import tokens, save_json
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.models.components import make_mask_infos

run, out = sys.argv[1], Path(sys.argv[2])
SEQ, MB = int(os.environ.get('EVAL_SEQ', '128')), 16
FRACTIONS = [float(x) for x in os.environ.get('EVAL_FRACTIONS', '0,.0001,.0003,.001,.003,.01,.03,.1,.2,.4,.6,.8,1').split(',')]
CONFIGS = os.environ.get('EVAL_CONFIGS', 'native,top0.005,top0.01,top0.1').split(',')
SCOPE = os.environ.get('EVAL_SCOPE', 'global')   # global: one r per piece for all rows; position: one per piece per position
RECORD = [int(x) for x in os.environ.get('EVAL_RECORD', '0,20,40,80,160,320').split(',')]
torch.backends.cuda.matmul.allow_tf32 = False
ri = ParamDecompRunInfo.from_path(run)
model = ComponentModel.from_run_info(ri).cuda().eval().requires_grad_(False)
MODULES = list(model.target_module_paths)
data, manifest = tokens('val')
ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)


def forward(ids, masks):
    with ac():
        return model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, masks)))).float()


def kl_sum(logits, out):
    logp = logits.float().log_softmax(-1)
    return (logp.exp() * (logp - out.log_softmax(-1))).sum()


started = time.perf_counter()
batches, sums = [], [dict(kl=0., ce=0., kept=0., n=0) for _ in FRACTIONS]
baseline_ce = 0.
for first in range(0, SEQ, MB):
    rows = torch.tensor(np.array(data[first:first+MB], dtype=np.int64), device='cuda')
    ids, labels = rows[:, :-1], rows[:, 1:]
    with torch.no_grad(), ac():
        base = model(ids, cache_type='input')
        ci = model.calc_causal_importances(pre_weight_acts=base.cache, detach_inputs=False, sampling='continuous').lower_leaky
        scores = [ci[n].float() for n in MODULES]; logits = base.output.float()
        del base, ci
    counts = [s.shape[-1] for s in scores]
    order = torch.cat(scores, -1).argsort(dim=-1, descending=True, stable=True)
    logp = logits.log_softmax(-1)
    baseline_ce += float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='sum'))
    with torch.no_grad():
        for fraction, total in zip(FRACTIONS, sums):
            k = min(sum(counts), round(fraction * sum(counts)))
            keep = torch.zeros_like(order, dtype=torch.float32).scatter_(-1, order[..., :k], 1.)
            pred = forward(ids, list(keep.split(counts, -1)))
            total['kl'] += float((logp.exp() * (logp - pred.log_softmax(-1))).sum())
            total['ce'] += float(F.cross_entropy(pred.flatten(0, 1), labels.flatten(), reduction='sum'))
            total['kept'] += k * ids.numel(); total['n'] += ids.numel()
            del pred, keep
    gates = {}
    for c in CONFIGS:
        if c == 'native': gates[c] = [s.clamp(0, 1).half() for s in scores]
        else:
            k = round(float(c[3:]) * sum(counts))
            gates[c] = list(torch.zeros(order.shape, dtype=torch.bool, device='cuda').scatter_(-1, order[..., :k], True).split(counts, -1))
    batches.append((ids, logits.bfloat16(), gates))
    del scores, order, logits, logp
curve = [dict(keep_fraction=f, kept_components=s['kept'] / s['n'], kl=s['kl'] / s['n'], ce=s['ce'] / s['n']) for f, s in zip(FRACTIONS, sums)]
n_tok = SEQ * 512
klkeep = dict(method='vpd', run=run, curve=curve, baseline_ce=baseline_ce / n_tok, positions=n_tok, total_components=sum(counts),
              delta_components_included=False, sensor='VPD CI (lower_leaky)', data_sha256=manifest['val']['sha256'],
              seconds=time.perf_counter() - started)
if SCOPE == 'global': save_json(klkeep, out/'klkeep.json'); print(json.dumps(dict(klkeep=curve)), flush=True)

result = {}
for c in CONFIGS:
    res = {}
    with torch.no_grad():
        res['hard_zero'] = sum(float(kl_sum(lg, forward(ids, [g.float() for g in gs[c]]))) for ids, lg, gs in batches) / n_tok
    if SCOPE == 'position':             # as adversarial_probe.py --scope position: each microbatch attacked on its own
        totals = {str(s): 0. for s in RECORD}
        for i, (ids, lg, gs) in enumerate(batches):
            gen = torch.Generator(device='cuda').manual_seed(i)
            adv = [torch.rand(*ids.shape, C, device='cuda', generator=gen).requires_grad_(True) for C in counts]
            for step in range(RECORD[-1] + 1):
                loss = kl_sum(lg, forward(ids, [g.float() + (1 - g.float()) * r for g, r in zip(gs[c], adv)]))
                if step in RECORD: totals[str(step)] += float(loss.detach())
                if step == RECORD[-1]: break
                grads = torch.autograd.grad(loss / ids.numel(), adv)
                with torch.no_grad():
                    for r, gr in zip(adv, grads): r.add_(0.1 * gr.sign()).clamp_(0., 1.)
                del loss, grads
        result[c] = {**res, **{k: v / n_tok for k, v in totals.items()}}
        print(json.dumps({c: result[c]}), flush=True)
        continue
    gen = torch.Generator(device='cuda').manual_seed(0)
    adv = [torch.rand(C, device='cuda', generator=gen).requires_grad_(True) for C in counts]
    for step in range(RECORD[-1] + 1):
        total, grads = 0., [torch.zeros_like(r) for r in adv]
        for ids, lg, gs in batches:
            loss = kl_sum(lg, forward(ids, [g.float() + (1 - g.float()) * r for g, r in zip(gs[c], adv)]))
            total += float(loss.detach())
            if step < RECORD[-1]:
                for acc, gr in zip(grads, torch.autograd.grad(loss / n_tok, adv)): acc += gr
            del loss
        if step in RECORD: res[str(step)] = total / n_tok
        if step == RECORD[-1]: break
        with torch.no_grad():
            for r, gr in zip(adv, grads): r.add_(0.1 * gr.sign()).clamp_(0., 1.)
    result[c] = res
    print(json.dumps({c: res}), flush=True)
save_json(dict(method='vpd', run=run, ranking='CI', scope=SCOPE, step_size=0.1, positions=n_tok, sequences=SEQ,
               data_sha256=manifest['val']['sha256'], kl=result, seconds=time.perf_counter() - started), out/f'adversarial_{SCOPE}_128.json')
print('VPD_SS_EVAL_DONE', flush=True)
