"""The 67M target (LlamaSimpleMLP: rotate-half RoPE, tanh GELU) as a plain PyTorch module, for running VPD's pieces and
the group programs on a laptop. Same code as matryoshka/coactive67.py. Base model: matryoshka/data/model_step_99999.pt."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET = Path(__file__).resolve().parents[1]/'matryoshka/data/model_step_99999.pt'

MODULES = tuple(f'h.{i}.{m}' for i in range(4) for m in
                ('attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'mlp.c_fc', 'mlp.down_proj'))
CFG = dict(vocab=50277, layers=4, heads=6, d=768, mlp=3072, eps=1e-6, base=10000., ctx=512)


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


def load_target(device='cpu'):
    m = Model(); sd = torch.load(TARGET, weights_only=True, map_location='cpu'); sd.setdefault('lm_head.weight', sd['wte.weight'])
    m.load_state_dict(sd); return m.to(device).eval().requires_grad_(False)
