"""Frozen-attention check of MAttr prompt circuits (prompt_mattr67.py): is a circuit's faithfulness carried by attention
patterns that only exist because the pieces outside it are off?

Per prompt and decomposition, the MAttr ranking is rebuilt exactly as prompt_mattr67.py (same seed, steps and budgets;
normal attention throughout training). Its hard top-k circuits (everything else off, VPD's delta off) are then scored on
the same grid twice: as before (attention recomputed from whatever the circuit leaves on), and with every layer's
attention pattern replaced by the intact model's on that prompt (the circuit's values and outputs go through the real
model's attention weights; its queries and keys no longer matter). Description length at each threshold both ways.
The target's attention runs its manual path (flash_attention false); each block is switched to the SDPA call, which is
replaced here by the same manual computation (mode None, identical ops, so the rankings reproduce prompt_mattr67.py),
by that computation with the pattern stored ('record'), or by the stored pattern ('frozen').
Usage: frozen_attention67.py --ours CKPT --vpd CKPT --output OUT.json [--methods ours vpd] [--shard i --shards n]
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from common import credentials, save_json, target
from prompt_mattr67 import GRID, K_MAX, STEPS, THRESHOLDS, Masked, curve, description_length, kl_last, load_factors, log, mattr, prompts

ACTIVE = None                                                # the Patterns object of the model running now


def manual_pattern(q, k):
    """The target's manual attention (llama_simple_mlp.CausalSelfAttention, flash_attention false), op for op."""
    T = q.shape[-2]
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1), float('-inf'))
    return F.softmax(att, dim=-1)


def _sdpa(q, k, v, *args, **kwargs):
    if ACTIVE is not None and ACTIVE.mode is not None: return ACTIVE.sdpa(q, k, v)
    return manual_pattern(q, k) @ v


F.scaled_dot_product_attention = _sdpa


class Patterns:
    """mode None: the target's own attention. 'record': causal softmax(q k^T / sqrt(d)) computed explicitly and stored per
    layer [1, heads, T, T]. 'frozen': the stored patterns, whatever q and k are."""

    def __init__(self, model):
        self.mode, self.layer, self.stored = None, None, {}
        def activate(*_):
            global ACTIVE
            ACTIVE = self
        model.register_forward_pre_hook(activate)
        for i, block in enumerate(model.h):
            assert not block.attn.flash_attention
            block.attn.flash_attention = True                       # route through the (replaced) SDPA call
            block.attn.register_forward_pre_hook(lambda _m, _a, i=i: setattr(self, 'layer', i))

    def sdpa(self, q, k, v):
        if self.mode == 'record': self.stored[self.layer] = manual_pattern(q, k).detach()
        return self.stored[self.layer].to(v.dtype) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', type=Path, required=True); ap.add_argument('--vpd', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--methods', nargs='+', default=['ours'], choices=['ours', 'vpd'])
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    all_prompts, val_sha = prompts()
    mine = list(range(args.shard, len(all_prompts), args.shards))[:args.limit]
    out = json.loads(args.output.read_text()) if args.output.exists() else dict(prompts={})
    out.update(grid=GRID, thresholds=list(THRESHOLDS), k_max=K_MAX, steps=STEPS, val_sha256=val_sha, methods=args.methods,
               procedure='prompt_mattr67 MAttr ranking (normal attention); hard top-k circuits scored with attention recomputed '
                         'and with every layer\'s attention pattern frozen to the intact model\'s')
    nets = {}
    for name in args.methods:
        model = target()
        factors, meta = load_factors('vpd' if name == 'vpd' else 'parauto', args.vpd if name == 'vpd' else args.ours, model)
        nets[name] = (Masked(model, factors), Patterns(model)); out[name] = meta
    for i in mine:
        if str(i) in out['prompts']: continue
        p = all_prompts[i]
        ids = torch.tensor([p['ids']], device='cuda'); T = ids.shape[1]
        entry = dict(text=p['text'], source=p['source'])
        for name, (net, pat) in nets.items():
            t0 = time.perf_counter()
            with torch.no_grad():
                intact_logp = net.logits(ids)[0, -1].log_softmax(-1)
                pat.mode = 'record'
                recorded_logp = net.logits(ids)[0, -1].log_softmax(-1)
                pat.mode = None
            label = int(intact_logp.argmax())
            init = net.gradient_attribution(ids, label).abs().clamp_min(1e-12).log()
            learned, _ = mattr(net, ids, init, intact_logp, STEPS, i, K_MAX, 'off')
            normal = curve(net, ids, learned, intact_logp, label, T, GRID, 'off', None)
            pat.mode = 'frozen'
            try:
                frozen = curve(net, ids, learned, intact_logp, label, T, GRID, 'off', None)
                with torch.no_grad():
                    ones = torch.ones(1, T, init.shape[1], device='cuda')
                    all_on_frozen = float(kl_last(intact_logp, net.logits(ids, ones))[0])
            finally:
                pat.mode = None
            dl = {r: {str(t): description_length(c, t) for t in THRESHOLDS} for r, c in (('normal', normal), ('frozen', frozen))}
            k5 = dl['normal']['0.5']
            at_k = lambda c, k: next((pt for pt in c if pt['k'] == k), None) if k is not None else None
            entry[name] = dict(label=label, p_label_intact=float(intact_logp[label].exp()),
                               recording_check_kl=float((intact_logp.exp() * (intact_logp - recorded_logp)).sum()),
                               all_pieces_on_frozen_kl=all_on_frozen, description_length=dl,
                               at_normal_k05=dict(k=k5, normal=at_k(normal, k5), frozen=at_k(frozen, k5)),
                               curves=dict(normal=normal, frozen=frozen), seconds=round(time.perf_counter() - t0, 1))
        out['prompts'][str(i)] = entry
        save_json(out, args.output)
        log(prompt=i, text=p['text'][-40:], **{n: dict(dl=entry[n]['description_length'], at_k05=entry[n]['at_normal_k05'],
                                                       check=round(entry[n]['recording_check_kl'], 5)) for n in nets})
    log(done=str(args.output), prompts=len(out['prompts']))


if __name__ == '__main__': main()
