"""Description length of prompt circuits by MAttr (Matryoshka attribution, arXiv 2609.25518) over (position, piece)
pairs, the same procedure for published VPD and our 13B run.

Per prompt and decomposition: one free score per (position, piece) pair, initialized at log |gradient attribution|
(activation x gradient: (h . r_c)(g . u_c), g the gradient of log p(y*) at the matrix output, y* the model's top token at
the last position). Each step draws B budgets (log-uniform on [1, K_MAX], stratified), keeps each budget's top-k pairs on
and every other piece off (the hard top-k in the forward pass, the sigmoid top-k Jacobian in the backward, as in
matryoshka67), and minimises the mean KL(intact || circuit) at the last position; Adam on the scores. The learned
ranking and the initial gradient ranking are then scored on a grid of k (hard top-k, KL at the last position and
p(y*)); the description length at threshold t is the smallest k on the grid with KL <= t. A circuit is the matrices'
pieces at full strength (VPD's delta W - V U left out, as in KLKeep); everything else in the model is unchanged.
Prompts: "The princess lost", then 10-token windows of held-out text (paired val rows ROWS, a random end position each,
the same draws as prompt_circuits67.py for its rows).
--regime vpd: pieces outside the circuit are not off but at a level r in [0, 1] (mask g + (1 - g) r, g = 1 in the
circuit), as VPD trains and evaluates. Random levels: r uniform and VPD's delta at a uniform level per position
(VPD's stochastic mask). Adversarial levels: r by PGD (random start, sign ascent on the KL, clamped to [0, 1]), delta
left out, one level per piece shared across the prompt's positions (the scope VPD trains and evaluates with; a level per
position breaks both decompositions until most pieces are kept). The scores are trained on the paper's
prompt-graph loss, half random levels and half adversarial (PGD_TRAIN), with budgets up to every (position, piece)
pair; the rankings are scored with the rest off, at random levels (mean of RANDOM_DRAWS) and at adversarial levels
(PGD_EVAL, VPD's adversarial eval).
Usage: prompt_mattr67.py --ours CKPT --vpd CKPT --output OUT.json [--regime off|vpd] [--shard i --shards n]
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

import numpy as np
import torch
import torch.nn.functional as F

from common import MODULES, TARGET_SHA, credentials, target, tokens as paired_tokens, save_json, sha

ROWS, PROMPT_LEN = range(1500, 1732), 10        # val rows no other eval here uses (val has 2,048; the first 32 = prompt_circuits67)
K_MAX, BUDGETS, STEPS, LR = 16384, 16, 400, 0.05
GRID = sorted({int(round(k)) for k in np.geomspace(1, K_MAX, 80)})
THRESHOLDS = (0.1, 0.5)
PGD_TRAIN, PGD_EVAL, RANDOM_DRAWS = (4, 1.0), (20, 0.1), 8      # (steps, step size); training: the paper's 4 steps, the app's step
VPD_GRID_POINTS = 90


def log(**fields): print(json.dumps(fields), flush=True)


def prompts():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    val, manifest = paired_tokens('val')
    rng = np.random.default_rng(0)
    out = [dict(source='paper', ids=tok.encode('The princess lost').ids)]
    for r in ROWS:
        end = int(rng.integers(PROMPT_LEN + 16, 512))
        out.append(dict(source=f'val row {r}, tokens {end - PROMPT_LEN}-{end - 1}', ids=[int(i) for i in val[r][end - PROMPT_LEN:end]]))
    for p in out: p['text'] = tok.decode(p['ids']); p['tokens'] = [tok.decode([i]) for i in p['ids']]
    return out, manifest['val']['sha256']


def load_factors(method, checkpoint, model):
    """(read [d_in, C], write [d_out, C]) per matrix, fp32 on the GPU."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), checkpoint_sha256=sha(checkpoint))
    if method == 'vpd':
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors'])
        factors = [(trainer.model.components[n].V.detach().float().clone(), trainer.model.components[n].U.detach().T.float().clone()) for n in MODULES]
        del trainer
    else:
        factors = [(p['read'].float().cuda(), p['write'].float().cuda()) for p in saved['factors']]
    return factors, meta


class Masked:
    """Forward hooks on the target: with masks set ([B, T, C] per matrix), each matrix outputs ((x R) * mask) U^T,
    plus (x D) * delta_level with delta levels set ([B, T, matrices]; D = W^T - R U^T, VPD's delta)."""

    def __init__(self, model, factors):
        self.model, self.factors, self.masks, self.delta_levels, self.capture = model, factors, None, None, None
        self.counts = [r.shape[1] for r, _ in factors]
        self.deltas = [model.get_submodule(n).weight.float().T - read @ write.T for n, (read, write) in zip(MODULES, factors)]
        for i, n in enumerate(MODULES):
            model.get_submodule(n).register_forward_hook(lambda _m, a, out, i=i: self.hook(i, a[0], out))

    def hook(self, i, x, out):
        if self.capture is not None:
            self.capture[i] = (x, out); out.retain_grad(); return None
        if self.masks is None: return None
        read, write = self.factors[i]
        y = ((x.float() @ read) * self.masks[i]) @ write.T
        if self.delta_levels is not None: y = y + (x.float() @ self.deltas[i]) * self.delta_levels[..., i, None]
        return y.to(out.dtype)

    def logits(self, ids, masks=None, delta_levels=None):
        self.masks = None if masks is None else list(masks.split(self.counts, -1))
        self.delta_levels = delta_levels
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16): return self.model(ids)[0].float()
        finally: self.masks = self.delta_levels = None

    def gradient_attribution(self, ids, label):
        """(h . r_c)(g . u_c) at every position for d log p(label) at the last position. [T, total]"""
        self.capture = {}
        wte = self.model.wte.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_())
        try:
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logp = self.model(ids)[0][0, -1].float().log_softmax(-1)[label]
            logp.backward()
            return torch.cat([((x[0].float() @ r) * (out.grad[0].float() @ u)).detach()
                              for (x, out), (r, u) in zip((self.capture[i] for i in range(len(MODULES))), self.factors)], -1)
        finally:
            wte.remove(); self.capture = None


class SigmoidTopk(torch.autograd.Function):
    """alpha = sigmoid(s + c_b), c_b per row by bisection so row b sums to k_b; implicit gradient (as matryoshka67)."""

    @staticmethod
    def forward(ctx, s, k):
        lo, hi = -s.max(1).values - 40, -s.min(1).values + 40
        for _ in range(60):
            mid = (lo + hi) / 2
            low = torch.sigmoid(s + mid[:, None]).sum(1) < k
            lo, hi = torch.where(low, mid, lo), torch.where(low, hi, mid)
        a = torch.sigmoid(s + ((lo + hi) / 2)[:, None])
        ctx.save_for_backward(a)
        return a

    @staticmethod
    def backward(ctx, g):
        a, = ctx.saved_tensors
        slope = a * (1 - a)
        return slope * (g - (slope * g).sum(1, keepdim=True) / slope.sum(1, keepdim=True).clamp_min(1e-30)), None


def hard_topk(s, ks):
    """0/1 masks [B, M] keeping the top k_b entries of one ranking s [M] (ties broken by position); ks [B]."""
    order = s.argsort(descending=True, stable=True)
    ranks = torch.empty_like(order); ranks[order] = torch.arange(len(order), device=s.device)
    return (ranks[None] < ks[:, None]).float()


def kl_last(intact_logp, logits):
    logp = logits[:, -1].log_softmax(-1)
    return (intact_logp.exp() * (intact_logp - logp)).sum(-1)


def outside(g, r):
    """Circuit pieces (g = 1) on, every other piece at level r."""
    return g + (1 - g) * r


def random_logits(net, ids, g, generator):
    """Pieces outside the circuit and VPD's delta at uniform random levels (VPD's stochastic mask)."""
    B, T, _ = g.shape
    r = torch.rand(g.shape, device='cuda', generator=generator)
    delta = torch.rand(B, T, len(MODULES), device='cuda', generator=generator)
    return net.logits(ids.expand(B, -1), outside(g, r), delta)


def adversary(net, ids, g, intact_logp, steps, step_size, generator):
    """Levels for the pieces outside the circuit that raise the KL: random start, sign ascent, clamped to [0, 1]
    (VPD's PGD; delta left out), one level per piece for all positions [B, 1, total]. g is detached."""
    r = torch.rand((g.shape[0], 1, g.shape[2]), device='cuda', generator=generator).requires_grad_(True)
    for _ in range(steps):
        with torch.enable_grad():
            loss = kl_last(intact_logp, net.logits(ids.expand(len(g), -1), outside(g, r))).sum()
            grad, = torch.autograd.grad(loss, r)
        with torch.no_grad(): r.add_(step_size * grad.sign()).clamp_(0, 1)
    return r.detach()


def curve(net, ids, scores, intact_logp, label, T, grid, regime, generator):
    """Hard top-k circuits of a ranking on the grid: KL at the last position and p(label), the rest off ('kl'), and
    with --regime vpd also at random levels ('kl_random', mean of RANDOM_DRAWS) and adversarial levels ('kl_adversarial')."""
    flat = scores.flatten()
    out = []
    for chunk in torch.tensor(grid, device='cuda').split(16):
        masks = hard_topk(flat, chunk).view(len(chunk), T, -1)
        with torch.no_grad():
            logits = net.logits(ids.expand(len(chunk), -1), masks)
        point = dict(kl=kl_last(intact_logp, logits), p_label=logits[:, -1].softmax(-1)[:, label])
        if regime == 'vpd':
            with torch.no_grad():
                draws = [random_logits(net, ids, masks, generator) for _ in range(RANDOM_DRAWS)]
            point['kl_random'] = torch.stack([kl_last(intact_logp, x) for x in draws]).mean(0)
            point['p_label_random'] = torch.stack([x[:, -1].softmax(-1)[:, label] for x in draws]).mean(0)
            r = adversary(net, ids, masks, intact_logp, *PGD_EVAL, generator)
            with torch.no_grad():
                logits = net.logits(ids.expand(len(chunk), -1), outside(masks, r))
            point.update(kl_adversarial=kl_last(intact_logp, logits), p_label_adversarial=logits[:, -1].softmax(-1)[:, label])
        out += [dict(k=int(k), **{n: float(v[j]) for n, v in point.items()}) for j, k in enumerate(chunk.tolist())]
    return out


def mattr(net, ids, init, intact_logp, steps, seed, k_max, regime):
    T, N = init.shape
    s = torch.nn.Parameter(init.clone())
    opt = torch.optim.Adam([s], lr=LR)
    g = torch.Generator(device='cuda').manual_seed(seed)
    lo, hi = 0., math.log(k_max)
    history = []
    for step in range(steps):
        u = (torch.arange(BUDGETS, device='cuda') + torch.rand(BUDGETS, device='cuda', generator=g)) / BUDGETS
        ks = torch.exp(lo + u * (hi - lo)).round().clamp(1, k_max)
        flat = s.view(1, -1).expand(BUDGETS, -1)
        alpha = SigmoidTopk.apply(flat, ks)
        hard = hard_topk(s.detach().flatten(), ks)                                # one sort: every row is the same ranking
        masks = (hard + alpha - alpha.detach()).view(BUDGETS, T, N)
        if regime == 'off':
            loss = kl_last(intact_logp, net.logits(ids.expand(BUDGETS, -1), masks)).mean()
        else:                                                                       # the paper's prompt-graph loss
            r = adversary(net, ids, masks.detach(), intact_logp, *PGD_TRAIN, g)
            loss = (kl_last(intact_logp, random_logits(net, ids, masks, g)).mean()
                    + kl_last(intact_logp, net.logits(ids.expand(BUDGETS, -1), outside(masks, r))).mean()) / 2
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1: history.append(dict(step=step, loss=float(loss.detach())))
    return s.detach(), history


def description_length(points, t, key='kl'):
    return next((p['k'] for p in points if p[key] <= t), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', type=Path, required=True); ap.add_argument('--vpd', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None, help='only the first N prompts of this shard (smoke tests)')
    ap.add_argument('--regime', choices=['off', 'vpd'], default='off')
    args = ap.parse_args(); credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    all_prompts, val_sha = prompts()
    mine = list(range(args.shard, len(all_prompts), args.shards))[:args.limit]
    out = json.loads(args.output.read_text()) if args.output.exists() else dict(prompts={})
    out.update(thresholds=list(THRESHOLDS), budgets_per_step=BUDGETS, steps=args.steps, lr=LR, val_sha256=val_sha,
               shard=args.shard, shards=args.shards, regime=args.regime)
    if args.regime == 'off':
        out.update(grid=GRID, k_max=K_MAX,
                   procedure='MAttr over (position, piece) pairs, init log|grad attribution|, hard top-k forward / sigmoid top-k backward, '
                             'KL at the last position, pieces outside the circuit off, VPD delta off')
    else:
        out.update(pgd_train=PGD_TRAIN, pgd_eval=PGD_EVAL, random_draws=RANDOM_DRAWS,
                   procedure='MAttr over (position, piece) pairs, init log|grad attribution|, hard top-k forward / sigmoid top-k backward, '
                             'KL at the last position; trained with pieces outside the circuit half at random levels (delta random), '
                             'half at PGD levels shared across positions (delta off), budgets log-uniform up to every pair; scored off / random / adversarial')
    nets = {}
    for name, ckpt in (('vpd', args.vpd), ('ours', args.ours)):
        model = target()
        factors, meta = load_factors('vpd' if name == 'vpd' else 'parauto', ckpt, model)
        nets[name] = Masked(model, factors); out[name] = meta
        if args.regime == 'vpd':
            pairs = PROMPT_LEN * sum(nets[name].counts)
            out[name].update(pieces=sum(nets[name].counts), grid=sorted({int(round(k)) for k in np.geomspace(1, pairs, VPD_GRID_POINTS)}))
    keys = ('kl',) if args.regime == 'off' else ('kl', 'kl_random', 'kl_adversarial')
    for i in mine:
        if str(i) in out['prompts']: continue
        p = all_prompts[i]
        ids = torch.tensor([p['ids']], device='cuda'); T = ids.shape[1]
        entry = dict(**p)
        for name, net in nets.items():
            t0 = time.perf_counter()
            with torch.no_grad():
                intact_logp = net.logits(ids)[0, -1].log_softmax(-1)
            label = int(intact_logp.argmax())
            attr = net.gradient_attribution(ids, label)
            init = attr.abs().clamp_min(1e-12).log()
            if args.regime == 'off': k_max, grid = K_MAX, GRID
            else: k_max = T * init.shape[1]; grid = sorted({min(k, k_max) for k in out[name]['grid']})
            learned, history = mattr(net, ids, init, intact_logp, args.steps, i, k_max, args.regime)
            g = torch.Generator(device='cuda').manual_seed(10_000 + i)
            curves = {r: curve(net, ids, s, intact_logp, label, T, grid, args.regime, g) for r, s in (('gradient', init), ('mattr', learned))}
            ones = torch.ones(1, T, init.shape[1], device='cuda')
            with torch.no_grad():
                all_on = float(kl_last(intact_logp, net.logits(ids, ones))[0])
                all_on_random = float(kl_last(intact_logp, random_logits(net, ids, ones, g))[0])
            dl = {r: {str(t): description_length(c, t) for t in THRESHOLDS} for r, c in curves.items()} if args.regime == 'off' else \
                {r: {key: {str(t): description_length(c, t, key) for t in THRESHOLDS} for key in keys} for r, c in curves.items()}
            entry[name] = dict(label=label, p_label_intact=float(intact_logp[label].exp()), kl_all_pieces_on=all_on, history=history,
                               curves=curves, description_length=dl, seconds=round(time.perf_counter() - t0, 1))
            if args.regime == 'vpd': entry[name]['kl_all_pieces_on_random_delta'] = all_on_random
        out['prompts'][str(i)] = entry
        save_json(out, args.output)
        log(prompt=i, text=p['text'][-40:], **{f'{n}_{r}': entry[n]['description_length'][r] for n in nets for r in ('gradient', 'mattr')},
            seconds={n: entry[n]['seconds'] for n in nets})
    log(done=str(args.output), prompts=len(out['prompts']))


if __name__ == '__main__': main()
