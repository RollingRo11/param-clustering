"""RESUMABLE SPD with the PAPER-EXACT gate input -- generated from
spd_toy_resumable.py. Sole change: the causal-importance MLPs receive
attn_output CONCATENATED with the position's own (abs-pos-encoded)
activation, per Christensen & Riggs eq. for x_bar_n:

    x_bar_n = (softmax(q_n K^T + r_n / sqrt(d_k)) V) (+) x_n

spd_toy.py's MatrixGate fed only the attention output (16-dim); here the
MLP input is 32-dim. Everything else is identical.

Original header follows.

RESUMABLE SPD -- generated from spd_toy.py; numerics unchanged.

Only difference from spd_toy.py: training runs in chunks of --chunk steps,
checkpointing (V, U, gates, Adam state, RNG state, step) to <out>/resume.pt
so a killed container costs one chunk instead of the whole run. The cosine
LR and the p-annealing are both functions of (step, args.steps), and
args.steps stays 100000 across every chunk, so the schedules are identical
to a single uninterrupted run.

Original docstring follows.

SPD on the toy induction model, matching Christensen & Riggs Smith
(arXiv:2511.08854) exactly.

Decomposition hyperparameters (their Table 4):
  steps 100,000 | batch 1,024 (global; DDP-split) | Adam lr 1e-3, cosine to 0
  C = 100 subcomponents per weight matrix (6 matrices -> 600 total)
  D_gate = 16 | last-token-only reconstruction
  losses: faithfulness beta=1000, recon beta=0.5, stochastic recon beta=1,
          stochastic LAYERWISE recon beta=1, importance minimality beta=0.02
          with L_p exponent annealed p: 0.9 -> 0.1

Their causal-importance function ("updated for sequential data"): per
decomposed matrix, a minimal attention network (1 head, 1 layer, QKV-only,
learned relative positional encodings on scores, learned absolute positional
encodings on the value vectors) mixes that matrix's input activations across
positions; independent per-subcomponent MLPs (hidden D_gate) map the mixed
representation to a scalar, through the leaky-hard sigmoid:
g_{c,n} = sigma_H(gamma_c(x_bar_n)).

SPD differences from VPD (vpd_toy.py): no adversarial PPGD, no weight-delta
spillover path in the component forward (masked forward uses (x@V * m)@U
alone; faithfulness pulls sum UV -> W), and the extra plain-recon +
layerwise-recon losses.

    python spd_toy.py --out out/spd_C600
"""
import argparse
import math
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from induction_model import InductionModel, gen_batch
from vpd_toy import MODULES, lower_leaky, upper_leaky

N_CTX = 64
D_MODEL = 16


class SPDLinear(nn.Module):
    """Replaces one nn.Linear. Component mode: (x @ V * mask) @ U — no
    weight-delta path (SPD; faithfulness pulls the sum toward W_target)."""

    def __init__(self, linear: nn.Linear, c: int):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.C = c
        self.register_buffer("W_target", linear.weight.detach().clone())
        self.V = nn.Parameter(torch.empty(d_in, c).normal_(0, 1 / math.sqrt(d_in)))
        self.U = nn.Parameter(torch.empty(c, d_out).normal_(0, 1 / math.sqrt(c)))
        self.mode = "target"
        self.mask = self.last_input = None

    def weight_delta(self):
        return self.W_target - (self.V @ self.U).T

    def forward(self, x):
        if self.mode == "target":
            self.last_input = x.detach()
            return F.linear(x, self.W_target)
        return ((x @ self.V) * self.mask) @ self.U


def install(model, c):
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = {}
    for path in MODULES:
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        wrappers[path] = SPDLinear(model.get_submodule(path), c)
        setattr(parent, attr, wrappers[path])
    return wrappers


class GateAttention(nn.Module):
    """Minimal attention for the CI function: 1 head, 1 layer, QKV-only,
    causal; learned relative positional bias on scores, learned absolute
    positional embeddings added to the values."""

    def __init__(self, d: int = D_MODEL, n_ctx: int = N_CTX):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.rel = nn.Parameter(torch.zeros(2 * n_ctx - 1))
        self.abs_pos = nn.Parameter(torch.zeros(n_ctx, d))
        self.d = d
        self.n_ctx = n_ctx

    def forward(self, x):
        n = x.shape[1]
        q, k = self.wq(x), self.wk(x)
        v = self.wv(x + self.abs_pos[:n])
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        idx = (torch.arange(n, device=x.device)[:, None]
               - torch.arange(n, device=x.device)[None, :]) + self.n_ctx - 1
        scores = scores + self.rel[idx]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool,
                                     device=x.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        return F.softmax(scores, dim=-1) @ v


class MatrixGate(nn.Module):
    """CI function for one matrix: GateAttention -> C independent MLPs
    (hidden D_gate) -> pre-sigmoid logits, batched via grouped weights."""

    def __init__(self, c: int, d: int = D_MODEL, d_gate: int = 16,
                 alpha: float = 0.01):
        super().__init__()
        self.attn = GateAttention(d)
        # paper: MLP input is attn_out (+) x_n  ->  width 2d
        self.w1 = nn.Parameter(torch.empty(c, 2 * d, d_gate).normal_(
            0, 1 / math.sqrt(2 * d)))
        self.b1 = nn.Parameter(torch.zeros(c, d_gate))
        self.w2 = nn.Parameter(torch.empty(c, d_gate).normal_(
            0, 1 / math.sqrt(d_gate)))
        self.b2 = nn.Parameter(torch.zeros(c))
        self.alpha = alpha

    def forward(self, x):
        n = x.shape[1]
        # x_bar_n = attn_out (+) (x_n + abs pos), per the paper
        xbar = torch.cat([self.attn(x),
                          x + self.attn.abs_pos[:n]], dim=-1)  # [B, S, 2d]
        h = F.gelu(torch.einsum("bsd,cdg->bscg", xbar, self.w1) + self.b1)
        logits = torch.einsum("bscg,cg->bsc", h, self.w2) + self.b2
        return lower_leaky(logits, self.alpha), upper_leaky(logits, self.alpha)


def faithfulness_loss(wrappers):
    sum_sq, numel = 0.0, 0
    for w in wrappers.values():
        d = w.weight_delta()
        sum_sq = sum_sq + d.pow(2).sum()
        numel += d.numel()
    return sum_sq / numel


def importance_minimality_loss(g_upper, p, eps=1e-12):
    """L_p: sum over subcomponents of E_{batch,pos}[(g + eps)^p]."""
    total = 0.0
    for g in g_upper.values():
        total = total + (g + eps).pow(p).mean(dim=(0, 1)).sum()
    return total


def kl_final(pred, target):
    log_q = F.log_softmax(pred[:, -1], -1)
    p = F.softmax(target[:, -1].detach(), -1)
    return F.kl_div(log_q, p, reduction="none").sum(-1).mean()


def masked_kl(model, wrappers, seq, target_logits, masks, subset=None):
    """KL at the final token with `subset` (default all) matrices replaced by
    their masked component sums; the rest stay target."""
    active = set(wrappers if subset is None else subset)
    for n, w in wrappers.items():
        if n in active:
            w.mode, w.mask = "component", masks[n]
    try:
        pred = model(seq)
    finally:
        for w in wrappers.values():
            w.mode, w.mask = "target", None
    return kl_final(pred, target_logits)


def anneal_p(step, total, p_start=0.9, p_end=0.1):
    return p_start + (p_end - p_start) * min(step / total, 1.0)


def worker(rank, world, args, rdv_file):
    dev = f"cuda:{rank}"
    if world > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method=f"file://{rdv_file}",
                                rank=rank, world_size=world)
    out_dir = args.out
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    torch.manual_seed(args.seed)   # identical init across ranks
    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    wrappers = install(model, args.c_per_module)
    model.to(dev)
    gates = nn.ModuleDict({n.replace(".", "_"): MatrixGate(args.c_per_module)
                           for n in MODULES}).to(dev)
    local_b = args.batch // world
    params = ([p for w in wrappers.values() for p in (w.V, w.U)]
              + list(gates.parameters()))
    log(f"{len(wrappers)} matrices x C={args.c_per_module}; gate params "
        f"{sum(p.numel() for p in gates.parameters()):,}; "
        f"global batch {args.batch} = {world} x {local_b}")

    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.gates = gates

        def forward(self, seq):
            for w in wrappers.values():
                w.mode, w.mask = "target", None
            target_logits = self.model(seq)
            g_lo, g_up = {}, {}
            for n in MODULES:
                lo, up = self.gates[n.replace(".", "_")](wrappers[n].last_input)
                g_lo[n], g_up[n] = lo, up
            return target_logits, g_lo, g_up

    cont = Container()
    ddp = (DistributedDataParallel(cont, device_ids=[rank]) if world > 1
           else cont)

    torch.manual_seed(args.seed + rank)   # diverge data/mask streams
    gen = torch.Generator(device=dev).manual_seed(args.seed + rank)
    opt = torch.optim.Adam(params, lr=args.lr)
    log_every = max(1, args.steps // 25)

    # -- resume ---------------------------------------------------------
    rpath = out_dir / "resume.pt"
    start = 0
    if rpath.exists():
        st = torch.load(rpath, map_location=dev, weights_only=False)
        for n, w in wrappers.items():
            w.V.data.copy_(st["wrappers"][n]["V"].to(dev))
            w.U.data.copy_(st["wrappers"][n]["U"].to(dev))
        gates.load_state_dict(st["gates"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"].cpu().to(torch.uint8))
        # stoch_masks use torch.rand_like -> the DEFAULT generator, not `gen`,
        # so the global streams must be restored too or the mask sequence
        # restarts every chunk.
        torch.set_rng_state(st["cpu_rng"].cpu().to(torch.uint8))
        torch.cuda.set_rng_state(st["cuda_rng"].cpu().to(torch.uint8), dev)
        start = int(st["step"])
        log(f"resumed at step {start}/{args.steps}")
    end = min(args.steps, start + (args.chunk or args.steps))

    for step in range(start, end):
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * step
                                           / max(args.steps - 1, 1)))
        for gg in opt.param_groups:
            gg["lr"] = lr
        seq, _, _ = gen_batch(local_b, dev, gen)

        target_logits, g_lo, g_up = ddp(seq)
        stoch_masks = {n: g + (1 - g) * torch.rand_like(g)
                       for n, g in g_lo.items()}
        ones = {n: torch.ones_like(g) for n, g in g_lo.items()}

        loss_faith = faithfulness_loss(wrappers)
        loss_recon = (torch.zeros((), device=dev) if args.paper_losses else
                      masked_kl(model, wrappers, seq, target_logits, ones))
        loss_stoch = masked_kl(model, wrappers, seq, target_logits,
                               stoch_masks)
        loss_layer = sum(
            masked_kl(model, wrappers, seq, target_logits, stoch_masks,
                      subset=[n]) for n in MODULES) / len(MODULES)
        loss_imp = importance_minimality_loss(
            g_up, anneal_p(step, args.steps))
        total = (1000.0 * loss_faith + 0.5 * loss_recon + 1.0 * loss_stoch
                 + 1.0 * loss_layer + 0.02 * loss_imp)

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        if step % log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                l0 = torch.stack([(g > 0.5).float().sum(-1).mean()
                                  for g in g_lo.values()]).sum()
            log(f"step {step:6d} faith {loss_faith:.3e} recon "
                f"{loss_recon:.4f} stoch {loss_stoch:.4f} layer "
                f"{loss_layer:.4f} imp {loss_imp:.3f} L0(g>.5) {l0:.1f}")

    if world > 1:
        dist.barrier()
    if rank != 0:
        dist.destroy_process_group()
        return

    torch.save({"step": end, "gen": gen.get_state().cpu(),
                "cpu_rng": torch.get_rng_state().cpu(),
                "cuda_rng": torch.cuda.get_rng_state(dev).cpu(),
                "opt": opt.state_dict(), "gates": gates.state_dict(),
                "wrappers": {n: {"V": w.V.detach().cpu(),
                                 "U": w.U.detach().cpu()}
                             for n, w in wrappers.items()}}, rpath)
    log(f"checkpointed at step {end}/{args.steps}")
    if end < args.steps:                 # more chunks to go; skip extraction
        if world > 1:
            dist.destroy_process_group()
        return

    # -- components = the subcomponents; per-matrix residual fold ------------
    with torch.no_grad():
        # mean gate activity on held-out data, for residual placement
        seq, _, _ = gen_batch(2048, dev, gen)
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        model(seq)
        total_g = {}
        for n in MODULES:
            lo, _ = gates[n.replace(".", "_")](wrappers[n].last_input)
            total_g[n] = lo.sum(dim=(0, 1))
        n_comp = args.c_per_module * len(MODULES)
        comps = {}
        for mi, path in enumerate(sorted(MODULES)):
            w = wrappers[path]
            sub = torch.einsum("co,ic->coi", w.U, w.V)
            dense = torch.zeros(n_comp, *w.W_target.shape, device=dev)
            sl = slice(mi * w.C, (mi + 1) * w.C)
            dense[sl] = sub
            dump = mi * w.C + int(total_g[path].argmin())
            dense[dump] += w.weight_delta()
            comps[path + ".weight"] = dense
            err = (dense.sum(0) - w.W_target).abs().max().item()
            assert err < 1e-4, (path, err)
        lab = torch.arange(n_comp, device=dev)
        alive = torch.cat([(total_g[n] > 1e-3).float()
                           for n in sorted(MODULES)])
        print(f"faith err {faithfulness_loss(wrappers):.3e}; subcomponents "
              f"with mean gate mass: {int(alive.sum())}/{n_comp}")

    torch.save({"components": {n: c.cpu() for n, c in comps.items()},
                "labels": lab.cpu(),
                "config": {k: str(v) for k, v in vars(args).items()}},
               out_dir / "components.pt")
    torch.save({"format": "spd", "c_per_module": args.c_per_module,
                "wrappers": {n: {"V": w.V.detach().cpu(),
                                 "U": w.U.detach().cpu()}
                             for n, w in wrappers.items()},
                "gates": gates.state_dict()}, out_dir / "spd_state.pt")
    print(f"saved {out_dir}/components.pt and spd_state.pt")
    if world > 1:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_per_module", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=0,
                    help="steps to run this invocation (0 = all)")
    ap.add_argument("--paper_losses", action="store_true",
                    help="drop the plain ones-mask recon (beta=0.5); the "
                         "paper's method section lists only four losses")
    ap.add_argument("--batch", type=int, default=1024, help="global batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "out/spd_C600")
    ap.add_argument("--world", type=int,
                    default=min(2, torch.cuda.device_count()))
    args = ap.parse_args()
    if args.world > 1:
        rdv = tempfile.NamedTemporaryFile(delete=False, suffix=".rdv")
        rdv.close()
        os.unlink(rdv.name)
        mp.spawn(worker, args=(args.world, args, rdv.name),
                 nprocs=args.world, join=True)
    else:
        worker(0, 1, args, "")


if __name__ == "__main__":
    main()
