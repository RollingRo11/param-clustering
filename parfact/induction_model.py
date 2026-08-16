"""Toy model of induction, following Christensen (2025), arXiv:2511.08854.

2-layer attention-only transformer, 1 head per layer, d_model=16, vocab 128,
seq len 64, Shortformer position encoding (positions enter queries/keys only,
never the residual stream), no LayerNorm, no MLP, no biases.

Task: sample n-2 tokens uniformly from the regular vocabulary, insert the
special s-token once at a random position and once at the end. The model is
trained (loss on the final position only) to predict the m-token: the token
that follows the first s-token. Solving it requires the classic previous-token
head -> induction head circuit.

Run as a script to train and save a checkpoint:
    python induction_model.py --out out/induction_model.pt
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 128          # last id is the special s-token
N_CTX = 64
D_MODEL = 16
N_LAYERS = 2
S_TOKEN = VOCAB - 1


class Attention(nn.Module):
    """Single-head causal attention; Shortformer positions go into q,k only."""

    def __init__(self, d: int):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.d = d

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        q = self.wq(x + pos)
        k = self.wk(x + pos)
        v = self.wv(x)
        n = x.shape[-2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        return self.wo(F.softmax(scores, dim=-1) @ v)


class InductionModel(nn.Module):
    def __init__(self, vocab: int = VOCAB, n_ctx: int = N_CTX,
                 d: int = D_MODEL, n_layers: int = N_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(n_ctx, d)
        self.layers = nn.ModuleList(Attention(d) for _ in range(n_layers))
        self.unembed = nn.Linear(d, vocab, bias=False)
        self.n_ctx = n_ctx

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        pos = self.pos.weight[: idx.shape[-1]]
        for layer in self.layers:
            x = x + layer(x, pos)
        return self.unembed(x)


def gen_batch(batch: int, device: torch.device,
              generator: torch.Generator | None = None,
              n_ctx: int = N_CTX, vocab: int = VOCAB):
    """Returns (seq [B, n_ctx], s_pos [B], target m-token [B])."""
    seq = torch.randint(0, vocab - 1, (batch, n_ctx), device=device,
                        generator=generator)
    # first s at p in [0, n-3] so the m-token at p+1 is a regular token
    s_pos = torch.randint(0, n_ctx - 2, (batch,), device=device,
                          generator=generator)
    rows = torch.arange(batch, device=device)
    seq[rows, s_pos] = S_TOKEN
    seq[:, -1] = S_TOKEN
    target = seq[rows, s_pos + 1]
    return seq, s_pos, target


def train_induction(model: InductionModel, steps: int = 100_000,
                    batch: int = 1024, lr: float = 1e-3,
                    weight_decay: float = 0.01, warmup: int = 1000,
                    seed: int = 0, log_every: int = 2000) -> float:
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warmup))
    model.train()
    for step in range(steps):
        seq, _, target = gen_batch(batch, device, gen)
        loss = F.cross_entropy(model(seq)[:, -1], target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
    return eval_induction(model, seed=seed + 1)


@torch.no_grad()
def eval_induction(model: InductionModel, batch: int = 4096,
                   seed: int = 1) -> float:
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    seq, _, target = gen_batch(batch, device, gen)
    model.eval()
    acc = (model(seq)[:, -1].argmax(-1) == target).float().mean().item()
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = InductionModel().to(args.device)
    acc = train_induction(model, steps=args.steps, batch=args.batch,
                          seed=args.seed)
    print(f"final-token accuracy: {acc:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "accuracy": acc,
                "steps": args.steps, "seed": args.seed}, args.out)
    print(f"saved {args.out}")
