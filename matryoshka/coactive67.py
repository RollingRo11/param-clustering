"""Which pieces of the 67M Matryoshka decomposition switch on together, within one matrix.

A piece is "on" (kept) at a token when it is in that token's top-k of all 43,008 pieces by the Fisher K=8 score
(as prompt_graph67.fisher_scores and the KLKeep eval: |h . read_c| * RMS over 8 sampled next tokens of (g . write_c)).

  score    python coactive67.py score --model M.pth --target T.pt --val val.uint16 --rows 128 --out DIR
           -> DIR/topk.pt: top-1290 piece ids per token (sorted by score), plus a KL check of the kept sets
  pairs    python coactive67.py pairs --out DIR
           -> DIR/pairs_k{K}.json: per matrix, pairs of pieces whose token sets nearly coincide (Jaccard >= 0.9)

Runs on a Mac (MPS) or CPU; the model is a plain re-implementation of LlamaSimpleMLP (rotate-half RoPE, tanh GELU).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODULES = tuple(f'h.{i}.{m}' for i in range(4) for m in
                ('attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'mlp.c_fc', 'mlp.down_proj'))
CFG = dict(vocab=50277, layers=4, heads=6, d=768, mlp=3072, eps=1e-6, base=10000., ctx=512)
KMAX = 1290                                         # largest budget kept per token (3% of 43,008)
BUDGETS = (129, 430, 1290)                          # 0.3%, 1%, 3% kept


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.square().mean(-1, keepdim=True) + CFG['eps'])


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj'): setattr(self, n, nn.Linear(CFG['d'], CFG['d'], bias=False))

    def forward(self, x, cos, sin):
        B, T, D = x.shape; H = CFG['heads']
        q, k, v = (p(x).view(B, T, H, D // H).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        rot = lambda t: t * cos + torch.cat((-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]), -1) * sin
        o = F.scaled_dot_product_attention(rot(q), rot(k), v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(CFG['d'], CFG['mlp'], bias=False); self.down_proj = nn.Linear(CFG['mlp'], CFG['d'], bias=False)

    def forward(self, x): return self.down_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.rms_1, self.attn, self.rms_2, self.mlp = RMSNorm(CFG['d']), Attn(), RMSNorm(CFG['d']), MLP()

    def forward(self, x, cos, sin):
        x = x + self.attn(self.rms_1(x), cos, sin)
        return x + self.mlp(self.rms_2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(CFG['vocab'], CFG['d']); self.h = nn.ModuleList(Block() for _ in range(CFG['layers']))
        self.ln_f = RMSNorm(CFG['d']); self.lm_head = nn.Linear(CFG['d'], CFG['vocab'], bias=False)
        hd = CFG['d'] // CFG['heads']
        f = torch.outer(torch.arange(CFG['ctx']).float(), 1. / CFG['base'] ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.cat((f, f), -1)
        self.register_buffer('cos', f.cos()[None, None], persistent=False); self.register_buffer('sin', f.sin()[None, None], persistent=False)

    def forward(self, idx):
        T = idx.shape[1]; x = self.wte(idx)
        for b in self.h: x = b(x, self.cos[:, :, :T], self.sin[:, :, :T])
        return self.lm_head(self.ln_f(x))


def load(target, model_path, device):
    m = Model(); sd = torch.load(target, map_location='cpu', weights_only=True)
    sd.setdefault('lm_head.weight', sd['wte.weight']); m.load_state_dict(sd, strict=True)
    m = m.to(device).eval().requires_grad_(False)
    saved = torch.load(model_path, map_location='cpu', weights_only=False, mmap=True)
    factors = [{k: f[k].float().to(device) for k in ('read', 'write')} for f in saved['factors']]
    err = max(float((f['write'] @ f['read'].T - m.get_submodule(n).weight).abs().max()) for n, f in zip(MODULES, factors))
    return m, factors, err


def fisher_scores(model, tokens, factors, k=8, seed=0):
    """[seq, 43008] Fisher K=k score, as prompt_graph67.fisher_scores (fp32 here)."""
    pre, post, hooks = {}, {}, []
    def keep(n):
        def hook(_m, a, o): pre[n], post[n] = a[0], o
        return hook
    for n in MODULES: hooks.append(model.get_submodule(n).register_forward_hook(keep(n)))
    hooks.append(model.wte.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_()))
    try:
        with torch.enable_grad(): logits = model(tokens)[0]
        g = torch.Generator(device='cpu').manual_seed(seed)
        samples = torch.multinomial(logits.detach().float().softmax(-1).cpu(), k, replacement=True, generator=g).to(logits.device)
        sq = [0.] * len(MODULES)
        for j in range(k):
            loss = F.cross_entropy(logits, samples[:, j], reduction='sum')
            grads = torch.autograd.grad(loss, [post[n] for n in MODULES], retain_graph=j < k - 1)
            for i, (gr, f) in enumerate(zip(grads, factors)): sq[i] = sq[i] + (gr[0] @ f['write']) ** 2
    finally:
        for h in hooks: h.remove()
    return torch.cat([(pre[n][0].detach() @ f['read']).abs() * (q / k).sqrt() for n, f, q in zip(MODULES, factors, sq)], -1), logits.detach()


def masked_logits(model, tokens, factors, keep):
    """Every matrix replaced by its kept pieces only: out = ((x @ read) * keep) @ write.T. keep [seq, 43008] bool."""
    masks = keep.float().split([f['read'].shape[1] for f in factors], -1); hooks = []
    for n, f, m in zip(MODULES, factors, masks):
        hooks.append(model.get_submodule(n).register_forward_hook(lambda _m, a, _o, f=f, m=m: ((a[0] @ f['read']) * m) @ f['write'].T))
    try: return model(tokens)[0]
    finally:
        for h in hooks: h.remove()


def score(a):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model, factors, err = load(a.target, a.model, device)
    val = np.memmap(a.val, mode='r', dtype=np.uint16).reshape(-1, 513)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ids, vals, checks = [], [], []
    for r in range(a.rows):
        tok = torch.from_numpy(val[r, :512].astype(np.int64))[None].to(device)
        s, logits = fisher_scores(model, tok, factors, seed=r)
        v, i = s.topk(KMAX, -1)
        ids.append(i.int().cpu()); vals.append(v.half().cpu())
        if r < a.check_rows:                                        # KL(intact || kept pieces only), mean over positions
            lp = logits.log_softmax(-1); row = dict(row=r, ce=float(F.cross_entropy(logits[:-1], tok[0, 1:])))
            for k in BUDGETS:
                keep = torch.zeros_like(s, dtype=torch.bool).scatter_(1, i[:, :k], True)
                lq = masked_logits(model, tok, factors, keep).log_softmax(-1)
                row[f'kl_{k}'] = float((lp.exp() * (lp - lq)).sum(-1).mean())
            checks.append(row); print(json.dumps(row), flush=True)
        if r % 8 == 0: print(json.dumps(dict(row=r, done=r + 1)), flush=True)
    torch.save(dict(ids=torch.cat(ids), scores=torch.cat(vals), counts=[f['read'].shape[1] for f in factors],
                    modules=MODULES, rows=a.rows, checks=checks, max_sum_error=err, model=str(a.model)), out/'topk.pt')


def pairs(a):
    out = Path(a.out); saved = torch.load(out/'topk.pt', weights_only=False)
    counts = saved['counts']; offsets = np.cumsum([0] + counts); N = saved['ids'].shape[0]
    for k in BUDGETS:
        ids = saved['ids'][:, :k].long()
        result = dict(budget=k, tokens=N, min_support=a.min_support, jaccard=a.jaccard, matrices={})
        for name, lo, hi in zip(MODULES, offsets[:-1], offsets[1:]):
            sel = (ids >= lo) & (ids < hi)                          # the kept pieces that live in this matrix
            B = torch.zeros(N, hi - lo)
            B[sel.nonzero(as_tuple=True)[0], ids[sel] - lo] = 1.
            n = B.sum(0)                                            # tokens each piece is on
            both = B.T @ B                                          # tokens both are on
            jac = both / (n[:, None] + n[None] - both).clamp_min(1)
            live = n >= a.min_support
            upper = torch.triu(torch.ones_like(jac, dtype=torch.bool), 1) & live[:, None] & live[None]
            # "nested": the rarer piece is (nearly) only on when the other is, P(other | rarer) >= jaccard
            rarer = torch.minimum(n[:, None], n[None]).clamp_min(1)
            nested = upper & (both / rarer >= a.jaccard)
            hit = upper & (jac >= a.jaccard)
            pa, pb = hit.nonzero(as_tuple=True)
            # groups: connected components of the high-Jaccard graph
            parent = list(range(hi - lo))
            def find(x):
                while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for x, y in zip(pa.tolist(), pb.tolist()): parent[find(x)] = find(y)
            groups = {}
            for x in set(pa.tolist()) | set(pb.tolist()): groups.setdefault(find(x), []).append(x)
            result['matrices'][name] = dict(
                pieces=int(hi - lo), live=int(live.sum()), always_on=int((n >= 0.5 * N).sum()),
                jaccard_pairs=int(hit.sum()), nested_pairs=int(nested.sum()),
                groups=sorted(([dict(pieces=sorted(g), support=[int(n[x]) for x in sorted(g)]) for g in groups.values()]),
                              key=lambda g: -len(g['pieces'])),
                top_pairs=[dict(a=int(x), b=int(y), n_a=int(n[x]), n_b=int(n[y]), both=int(both[x, y]), jaccard=round(float(jac[x, y]), 4))
                           for x, y in sorted(zip(pa.tolist(), pb.tolist()), key=lambda p: -float(both[p[0], p[1]]))[:50]])
        (out/f'pairs_k{k}.json').write_text(json.dumps(result, indent=1))
        print(json.dumps(dict(budget=k, **{m: {x: v[x] for x in ('live', 'always_on', 'jaccard_pairs', 'nested_pairs')}
                                         for m, v in result['matrices'].items()})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score'); s.add_argument('--model'); s.add_argument('--target'); s.add_argument('--val')
    s.add_argument('--rows', type=int, default=128); s.add_argument('--check-rows', type=int, default=4); s.add_argument('--out')
    q = sub.add_parser('pairs'); q.add_argument('--out'); q.add_argument('--min-support', type=int, default=50)
    q.add_argument('--jaccard', type=float, default=0.9)
    a = p.parse_args(); dict(score=score, pairs=pairs)[a.cmd](a)
