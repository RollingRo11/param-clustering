"""Description length of prompt circuits, measured the way our decomposition is trained: pieces outside the circuit are off,
and the circuit must reproduce the intact model's whole next-token distribution at the last position.

For every prompt and both decompositions (published VPD, our 13B run), VPD's own per-prompt optimization
(param_decomp.app.backend.optim_cis.optimize_ci_values) with mask_type 'ci' (the CI is the mask, so pieces outside the
circuit are off), the KL loss at the last position (KLLossConfig), no PGD, importance minimality p = 0.3, 2,000 steps,
CI learning rate 0.01 (the app's), over a sweep of the importance-minimality coefficient. Ours enters as in prompt_graph67.py (rank-1 components; at
each position the alive set is our top-n pieces by Fisher K=8 score, n = VPD's alive count there).
Per run: nodes (CI > 0) and the KL to the intact model at the last position with only the circuit's pieces at full
strength (everything else off; VPD's delta left out), plus the KL with the CI as the mask.
Description length of a prompt at threshold t: the fewest nodes among the sweep's circuits with that KL <= t.
Prompts: "The princess lost", then 10-token windows of held-out text (paired val rows ROWS, a random end position each).
Usage: prompt_circuits67.py --ours CKPT --vpd CKPT --output OUT.json [--shard i --shards n]
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

from common import MODULES, credentials, target, tokens as paired_tokens, save_json
from prompt_graph67 import CI_LR, load_ours, load_vpd, our_initial_ci

ROWS, PROMPT_LEN, N_PROMPTS = range(1500, 1532), 10, 32           # val rows no other eval here uses (val has 2,048)
SWEEP = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
THRESHOLDS = (0.1, 0.5)


def log(**fields): print(json.dumps(fields), flush=True)


def prompts():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    val, manifest = paired_tokens('val')
    rng = np.random.default_rng(0)
    out = [dict(source='paper', ids=tok.encode('The princess lost').ids)]
    for r in ROWS[:N_PROMPTS]:
        end = int(rng.integers(PROMPT_LEN + 16, 512))
        out.append(dict(source=f'val row {r}, tokens {end - PROMPT_LEN}-{end - 1}', ids=[int(i) for i in val[r][end - PROMPT_LEN:end]]))
    for p in out: p['text'] = tok.decode(p['ids']); p['tokens'] = [tok.decode([i]) for i in p['ids']]
    return out, manifest['val']['sha256']


def circuit(cm, cfg, tokens, coeff, steps, seed=0):
    from param_decomp.app.backend.optim_cis import KLLossConfig, OptimCIConfig, optimize_ci_values
    from param_decomp.configs import ImportanceMinimalityLossConfig
    from param_decomp.models.components import make_mask_infos
    pos = tokens.shape[1] - 1
    config = OptimCIConfig(seed=seed, lr=CI_LR, steps=steps, weight_decay=0.0, lr_schedule='cosine', lr_exponential_halflife=None,
                           lr_warmup_pct=0.01, log_freq=max(1, steps // 4),
                           imp_min_config=ImportanceMinimalityLossConfig(coeff=coeff, pnorm=0.3, beta=0.0),
                           loss_config=KLLossConfig(coeff=1.0, position=pos), sampling=cfg.sampling, ce_kl_rounding_threshold=0.5,
                           mask_type='ci', adv_pgd=None)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    result = optimize_ci_values(model=cm, tokens=tokens, config=config, device='cuda')
    ci = result.params.create_ci_outputs(cm, 'cuda').lower_leaky
    with torch.no_grad():
        intact = F.log_softmax(cm(tokens)[0, pos].float(), -1)
        kl = lambda masks: float((intact.exp() * (intact - F.log_softmax(cm(tokens, mask_infos=make_mask_infos(masks))[0, pos].float(), -1))).sum())
        kl_binary, kl_ci = kl({n: (v > 0).float() for n, v in ci.items()}), kl(ci)
    return dict(imp_min_coeff=coeff, nodes=int(sum((v > 0).sum() for v in ci.values())),
                nodes_per_position=[int(sum((ci[n][0, t] > 0).sum() for n in MODULES)) for t in range(tokens.shape[1])],
                kl_binary=kl_binary, kl_ci_mask=kl_ci, seconds=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', type=Path, required=True); ap.add_argument('--vpd', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    all_prompts, val_sha = prompts()
    mine = list(range(args.shard, len(all_prompts), args.shards))
    out = json.loads(args.output.read_text()) if args.output.exists() else dict(prompts={})
    out.update(sweep=list(SWEEP), thresholds=list(THRESHOLDS), steps=args.steps, val_sha256=val_sha, shard=args.shard, shards=args.shards,
               procedure=f'optimize_ci_values, mask_type ci, KL loss at the last position, no PGD, p 0.3, beta 0, lr {CI_LR} cosine')
    vpd_model, ours_model = target(), target()
    vpd, cfg, vmeta = load_vpd(vpd_model, args.vpd)
    ours, factors, ometa = load_ours(ours_model, args.ours, cfg)
    out.update(vpd=vmeta, ours=ometa)
    for i in mine:
        p = all_prompts[i]
        if str(i) in out['prompts'] and len(out['prompts'][str(i)].get('ours', [])) == len(SWEEP): continue
        tokens = torch.tensor([p['ids']], device='cuda')
        with torch.no_grad():
            ci0 = vpd.calc_causal_importances(pre_weight_acts=vpd(tokens, cache_type='input').cache, sampling=cfg.sampling, detach_inputs=False)
        alive = [int(sum((ci0.lower_leaky[n][0, t] > 0).sum() for n in MODULES)) for t in range(len(p['ids']))]
        our_initial_ci(ours, ours_model, factors, tokens, None, alive)
        entry = dict(**p, alive_per_position=alive, vpd=[], ours=[])
        for name, cm in (('vpd', vpd), ('ours', ours)):
            for coeff in SWEEP:
                r = circuit(cm, cfg, tokens, coeff, args.steps)
                entry[name].append(r)
                log(prompt=i, method=name, **{k: v for k, v in r.items() if k != 'nodes_per_position'})
        for name in ('vpd', 'ours'):
            entry[f'{name}_description_length'] = {str(t): min((r['nodes'] for r in entry[name] if r['kl_binary'] <= t), default=None)
                                                   for t in THRESHOLDS}
        out['prompts'][str(i)] = entry
        save_json(out, args.output)
        log(prompt=i, text=p['text'], vpd=entry['vpd_description_length'], ours=entry['ours_description_length'])
    log(done=str(args.output))


if __name__ == '__main__': main()
