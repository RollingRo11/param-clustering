"""SPD for the multitask sparse parity MLP (msp_model.py checkpoint).

Same recipe as our Table-4-faithful induction replication
(spd_toy_resumable.py): per-matrix rank-1 subcomponents x@V*mask@U, per-
component MLP gates on the wrapped layer's input, five losses (faithfulness,
recon KL beta=0.5, stochastic KL, per-layer stochastic KL, Lp importance
minimality with p annealed 0.9->0.1), chunked resume via <out>/resume.pt,
optional W&B (no-op without WANDB_API_KEY).

Differences from the induction version: no sequence axis (gates are plain
per-component MLPs on the layer input), and the MLP has biases -- biases are
NOT decomposed; they stay fixed and are added in every mode.

    python spd_msp.py --ckpt out/msp/model.pt --out out/msp/spd_C300
"""
import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from msp_model import MSPModel, make_tasks, sample_batch, N_TASKS, N_BITS, K
from vpd_toy import lower_leaky, upper_leaky

MODULES = ("fc1", "fc2")


class SPDLinear(nn.Module):
    """Replaces one nn.Linear. Component mode: ((x @ V) * mask) @ U + b.
    The bias is not decomposed: it is frozen and applied in every mode."""

    def __init__(self, linear: nn.Linear, c: int):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.C = c
        self.register_buffer("W_target", linear.weight.detach().clone())
        self.register_buffer("bias", linear.bias.detach().clone())
        self.V = nn.Parameter(torch.empty(d_in, c).normal_(
            0, 1 / math.sqrt(d_in)))
        self.U = nn.Parameter(torch.empty(c, d_out).normal_(
            0, 1 / math.sqrt(c)))
        self.mode = "target"
        self.mask = self.last_input = None

    def weight_delta(self):
        return self.W_target - (self.V @ self.U).T

    def forward(self, x):
        if self.mode == "target":
            self.last_input = x.detach()
            return F.linear(x, self.W_target, self.bias)
        return ((x @ self.V) * self.mask) @ self.U + self.bias


class MatrixGate(nn.Module):
    """C independent MLPs (hidden d_gate) on the layer input -> pre-sigmoid
    logits, batched via grouped weights. No attention: MSP has no sequence."""

    def __init__(self, c: int, d_in: int, d_gate: int = 16,
                 alpha: float = 0.01):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(c, d_in, d_gate).normal_(
            0, 1 / math.sqrt(d_in)))
        self.b1 = nn.Parameter(torch.zeros(c, d_gate))
        self.w2 = nn.Parameter(torch.empty(c, d_gate).normal_(
            0, 1 / math.sqrt(d_gate)))
        self.b2 = nn.Parameter(torch.zeros(c))
        self.alpha = alpha

    def forward(self, x):
        h = F.gelu(torch.einsum("bd,cdg->bcg", x, self.w1) + self.b1)
        logits = torch.einsum("bcg,cg->bc", h, self.w2) + self.b2
        return lower_leaky(logits, self.alpha), upper_leaky(logits, self.alpha)


def install(model, c):
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = {}
    for path in MODULES:
        wrappers[path] = SPDLinear(getattr(model, path), c)
        setattr(model, path, wrappers[path])
    return wrappers


def faithfulness_loss(wrappers):
    sum_sq, numel = 0.0, 0
    for w in wrappers.values():
        d = w.weight_delta()
        sum_sq = sum_sq + d.pow(2).sum()
        numel += d.numel()
    return sum_sq / numel


def importance_minimality_loss(g_upper, p, eps=1e-12):
    total = 0.0
    for g in g_upper.values():
        total = total + (g + eps).pow(p).mean(dim=0).sum()
    return total


def kl_out(pred, target):
    log_q = F.log_softmax(pred, -1)
    p = F.softmax(target.detach(), -1)
    return F.kl_div(log_q, p, reduction="none").sum(-1).mean()


def masked_kl(model, wrappers, x, target_logits, masks, subset=None):
    active = set(wrappers if subset is None else subset)
    for n, w in wrappers.items():
        if n in active:
            w.mode, w.mask = "component", masks[n]
    try:
        pred = model(x)
    finally:
        for w in wrappers.values():
            w.mode, w.mask = "target", None
    return kl_out(pred, target_logits)


def anneal_p(step, total, p_start=0.9, p_end=0.1):
    return p_start + (p_end - p_start) * min(step / total, 1.0)


def wandb_run(args, out_dir):
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "param-clustering"),
        id=f"{out_dir.parent.name}-{out_dir.name}",
        name=f"{out_dir.parent.name}-{out_dir.name}", resume="allow",
        dir=str(out_dir),
        config={k: str(v) for k, v in vars(args).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c_per_module", type=int, default=150)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=0,
                    help="steps to run this invocation (0 = all)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/msp/model.pt")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "out/msp/spd_C300")
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    torch.manual_seed(args.seed)
    model = MSPModel(width=int(ck["config"]["width"])).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    Ss = [torch.tensor(s) for s in ck["Ss"]]
    probs = torch.tensor(ck["probs"])

    wrappers = install(model, args.c_per_module)
    d_ins = {"fc1": N_TASKS + N_BITS, "fc2": int(ck["config"]["width"])}
    gates = nn.ModuleDict({n: MatrixGate(args.c_per_module, d_ins[n])
                           for n in MODULES}).to(dev)
    params = ([p for w in wrappers.values() for p in (w.V, w.U)]
              + list(gates.parameters()))
    print(f"{len(wrappers)} matrices x C={args.c_per_module}; gate params "
          f"{sum(p.numel() for p in gates.parameters()):,}; "
          f"batch {args.batch}", flush=True)

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    opt = torch.optim.Adam(params, lr=args.lr)
    log_every = max(1, args.steps // 25)
    run = wandb_run(args, out_dir)

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
        torch.set_rng_state(st["cpu_rng"].cpu().to(torch.uint8))
        if dev.startswith("cuda"):
            torch.cuda.set_rng_state(st["cuda_rng"].cpu().to(torch.uint8), dev)
        start = int(st["step"])
        print(f"resumed at step {start}/{args.steps}", flush=True)
    end = min(args.steps, start + (args.chunk or args.steps))

    for step in range(start, end):
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * step
                                           / max(args.steps - 1, 1)))
        for gg in opt.param_groups:
            gg["lr"] = lr
        x, _, _ = sample_batch(args.batch, Ss, probs, N_TASKS, N_BITS, dev,
                               gen)
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        target_logits = model(x)
        g_lo, g_up = {}, {}
        for n in MODULES:
            lo, up = gates[n](wrappers[n].last_input)
            g_lo[n], g_up[n] = lo, up
        stoch_masks = {n: g + (1 - g) * torch.rand_like(g)
                       for n, g in g_lo.items()}
        ones = {n: torch.ones_like(g) for n, g in g_lo.items()}

        loss_faith = faithfulness_loss(wrappers)
        loss_recon = masked_kl(model, wrappers, x, target_logits, ones)
        loss_stoch = masked_kl(model, wrappers, x, target_logits, stoch_masks)
        loss_layer = sum(
            masked_kl(model, wrappers, x, target_logits, stoch_masks,
                      subset=[n]) for n in MODULES) / len(MODULES)
        loss_imp = importance_minimality_loss(g_up,
                                              anneal_p(step, args.steps))
        total = (1000.0 * loss_faith + 0.5 * loss_recon + 1.0 * loss_stoch
                 + 1.0 * loss_layer + 0.02 * loss_imp)

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        if run is not None and (step % 1000 == 0 or step == args.steps - 1):
            with torch.no_grad():
                l0 = torch.stack([(g > 0.5).float().sum(-1).mean()
                                  for g in g_lo.values()]).sum()
            run.log({"faith": loss_faith.item(), "recon": loss_recon.item(),
                     "stoch": loss_stoch.item(), "layer": loss_layer.item(),
                     "imp": loss_imp.item(), "total": total.item(),
                     "lr": lr, "L0_gt0.5": l0.item()}, step=step)

        if step % log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                l0 = torch.stack([(g > 0.5).float().sum(-1).mean()
                                  for g in g_lo.values()]).sum()
            print(f"step {step:6d} faith {loss_faith:.3e} recon "
                  f"{loss_recon:.4f} stoch {loss_stoch:.4f} layer "
                  f"{loss_layer:.4f} imp {loss_imp:.3f} L0(g>.5) {l0:.1f}",
                  flush=True)

    torch.save({"step": end, "gen": gen.get_state().cpu(),
                "cpu_rng": torch.get_rng_state().cpu(),
                "cuda_rng": (torch.cuda.get_rng_state(dev).cpu()
                             if dev.startswith("cuda") else
                             torch.get_rng_state().cpu()),
                "opt": opt.state_dict(), "gates": gates.state_dict(),
                "wrappers": {n: {"V": w.V.detach().cpu(),
                                 "U": w.U.detach().cpu()}
                             for n, w in wrappers.items()}}, rpath)
    print(f"checkpointed at step {end}/{args.steps}", flush=True)
    if end < args.steps:
        if run is not None:
            run.finish()
        return

    # -- components = the subcomponents; per-matrix residual fold ----------
    with torch.no_grad():
        x, _, _ = sample_batch(8192, Ss, probs, N_TASKS, N_BITS, dev, gen)
        for w in wrappers.values():
            w.mode, w.mask = "target", None
        model(x)
        total_g = {n: gates[n](wrappers[n].last_input)[0].sum(0)
                   for n in MODULES}
        n_comp = args.c_per_module * len(MODULES)
        comps = {}
        for mi, path in enumerate(MODULES):
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
        alive = torch.cat([(total_g[n] / 8192 > 1e-3).float()
                           for n in MODULES])
        print(f"faith err {faithfulness_loss(wrappers):.3e}; subcomponents "
              f"with mean gate mass: {int(alive.sum())}/{n_comp}")

    torch.save({"components": {n: c.cpu() for n, c in comps.items()},
                "config": {k: str(v) for k, v in vars(args).items()}},
               out_dir / "components.pt")
    torch.save({"format": "spd", "c_per_module": args.c_per_module,
                "wrappers": {n: {"V": w.V.detach().cpu(),
                                 "U": w.U.detach().cpu()}
                             for n, w in wrappers.items()},
                "gates": gates.state_dict()}, out_dir / "spd_state.pt")
    print(f"saved {out_dir}/components.pt and spd_state.pt")
    if run is not None:
        import wandb
        art = wandb.Artifact(out_dir.name, type="spd-decomposition")
        art.add_file(str(out_dir / "spd_state.pt"))
        art.add_file(str(out_dir / "components.pt"))
        run.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    main()
