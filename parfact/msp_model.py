"""Multitask sparse parity (MSP) model, following Michaud et al. 2023
(arXiv:2303.13506, "The Quantization Model of Neural Scaling"), matching the
defaults of their scripts/sparse-parity-v4.py.

Input is [n_tasks one-hot task code | n random bits]; the label is the parity
of the k task-specific bits S_i. Task i is sampled with probability
p_i ~ i^-alpha (power law), so the model learns per-task "quanta" in
frequency order -- the ground-truth skill distribution our decompositions
should recover.

Defaults (theirs): n_tasks=100, n=50, k=3, alpha=1.5, width=100,
depth=2 (fc1: 150->100, ReLU, fc2: 100->2), Adam lr 1e-3, batch 10,000,
25k steps, infinite data, cross-entropy.

    python msp_model.py --out out/msp/model.pt
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N_TASKS, N_BITS, K = 100, 50, 3
ALPHA = 1.5
WIDTH = 100


class MSPModel(nn.Module):
    def __init__(self, n_tasks: int = N_TASKS, n_bits: int = N_BITS,
                 width: int = WIDTH):
        super().__init__()
        self.fc1 = nn.Linear(n_tasks + n_bits, width)
        self.fc2 = nn.Linear(width, 2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def make_tasks(n_tasks: int, n_bits: int, k: int, seed: int):
    """Fixed k-bit subsets S_i and power-law sampling probabilities."""
    g = torch.Generator().manual_seed(seed)
    Ss = [torch.randperm(n_bits, generator=g)[:k].sort().values
          for _ in range(n_tasks)]
    probs = torch.arange(1, n_tasks + 1, dtype=torch.float64).pow(-ALPHA)
    return Ss, (probs / probs.sum()).float()


def sample_batch(bsz, Ss, probs, n_tasks, n_bits, dev, gen,
                 tasks=None):
    """Inputs [bsz, n_tasks+n_bits], labels, task codes. `tasks` overrides
    the power-law draw (used for per-task eval)."""
    if tasks is None:
        tasks = torch.multinomial(probs.to(dev), bsz, replacement=True,
                                  generator=gen)
    bits = torch.randint(0, 2, (bsz, n_bits), device=dev, generator=gen,
                         dtype=torch.float32)
    S = torch.stack([Ss[t] for t in tasks.tolist()]).to(dev)
    y = bits.gather(1, S).sum(1).long() % 2
    code = F.one_hot(tasks, n_tasks).float()
    return torch.cat([code, bits], dim=1), y, tasks


def per_task_accuracy(model, Ss, probs, n_tasks, n_bits, dev, gen,
                      points_per_task=1000):
    accs = []
    with torch.no_grad():
        for t in range(n_tasks):
            tt = torch.full((points_per_task,), t, device=dev)
            x, y, _ = sample_batch(points_per_task, Ss, probs, n_tasks,
                                   n_bits, dev, gen, tasks=tt)
            accs.append((model(x).argmax(-1) == y).float().mean().item())
    return accs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=25_000)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--batch", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "out/msp/model.pt")
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    Ss, probs = make_tasks(N_TASKS, N_BITS, K, args.seed)
    model = MSPModel(width=args.width).to(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            run = wandb.init(project=os.environ.get("WANDB_PROJECT",
                                                    "param-clustering"),
                             id=args.out.parent.name + "-train",
                             name=args.out.parent.name + "-train",
                             resume="allow",
                             dir=str(args.out.parent),
                             config=vars(args) | {"n_tasks": N_TASKS,
                                                  "n": N_BITS, "k": K,
                                                  "alpha": ALPHA})
        except ImportError:
            pass

    for step in range(args.steps):
        x, y, _ = sample_batch(args.batch, Ss, probs, N_TASKS, N_BITS, dev,
                               gen)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            acc = (model(x).argmax(-1) == y).float().mean().item()
            print(f"step {step:6d} loss {loss.item():.4f} "
                  f"train-dist acc {acc:.4f}", flush=True)
            if run is not None:
                run.log({"loss": loss.item(), "acc": acc}, step=step)

    accs = per_task_accuracy(model, Ss, probs, N_TASKS, N_BITS, dev, gen)
    n_learned = sum(a > 0.9 for a in accs)
    print(f"tasks with acc>0.9: {n_learned}/{N_TASKS}")
    print("acc by task decile:",
          [round(sum(accs[i:i + 10]) / 10, 3) for i in range(0, 100, 10)])

    torch.save({"state_dict": model.state_dict(),
                "Ss": [s.tolist() for s in Ss],
                "probs": probs.tolist(),
                "per_task_acc": accs,
                "config": {"n_tasks": N_TASKS, "n": N_BITS, "k": K,
                           "alpha": ALPHA, "width": args.width,
                           **{k: str(v) for k, v in vars(args).items()}}},
               args.out)
    print("saved", args.out)
    if run is not None:
        run.log({"tasks_learned": n_learned})
        run.finish()


if __name__ == "__main__":
    main()
