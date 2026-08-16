"""VPD on the toy induction model, adapted from spd_lib/nano_param_decomp/run.py.

Faithful single-GPU miniature of the nano reference implementation (same
method sections, toy-scaled): each of the 8 attention Linears is replaced by a
ComponentLinear holding C=100 rank-one subcomponents (V [d_in,C], U [C,d_out]);
a small shared CI transformer maps pre-weight activations to per-subcomponent
causal importances through leaky-hard sigmoids; training minimizes

    coeff_faith * ||W - (VU)^T||^2 / numel        (+ 400-step warmup)
  + coeff_imp   * L_p importance minimality (p annealed 2.0 -> 0.4)
  + coeff_stoch * KL under stochastic masks  m = ci + (1-ci) U(0,1)
  + coeff_ppgd  * KL under persistent-PGD adversarial masks

Reconstruction KL is taken at the final position (the induction prediction --
the behavior the task defines and the ablation curve measures); masks at every
position still matter through attention.

VPD's native units are per-matrix rank-one subcomponents; full model-spanning
parameter components come from clustering causal-importance profiles (paper
app: clustering). Here the 800 subcomponents are k-means clustered on their
normalized CI profiles into --n_components=100 components; the (tiny)
faithfulness residual is folded into the least-used cluster so the components
sum to the target weights exactly. Output is consumable by
ablation_curve.py --components.

    python vpd_toy.py                  # train + cluster + save
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from induction_model import InductionModel, gen_batch
from prev_method import kmeans

MODULES = tuple(f"layers.{l}.{m}" for l in range(2)
                for m in ("wq", "wk", "wv", "wo"))


# --- leaky-hard sigmoids (nano section B) ---

class _LowerLeakyHardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        zero = torch.zeros_like(grad_output)
        grad = torch.where(
            x <= 0,
            torch.where(grad_output < 0, ctx.alpha * grad_output, zero),
            torch.where(x <= 1, grad_output, zero))
        return grad, None


def lower_leaky(x, alpha):
    return _LowerLeakyHardSigmoid.apply(x, alpha)


def upper_leaky(x, alpha):
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0.0, 1.0))


# --- ComponentLinear (nano section C) ---

class ComponentLinear(nn.Module):
    """Replaces one nn.Linear; W_target ~= (V @ U).T, masked per position."""

    def __init__(self, linear: nn.Linear, c: int):
        super().__init__()
        d_out, d_in = linear.weight.shape
        self.C = c
        self.register_buffer("W_target", linear.weight.detach().clone())
        self.V = nn.Parameter(torch.empty(d_in, c).normal_(0, 1 / math.sqrt(d_in)))
        self.U = nn.Parameter(torch.empty(c, d_out).normal_(0, 1 / math.sqrt(c)))
        self.mode = "target"
        self.mask = self.delta_mask = self.last_input = None

    def weight_delta(self):
        return self.W_target - (self.V @ self.U).T

    def forward(self, x):
        if self.mode == "target":
            self.last_input = x.detach()
            return F.linear(x, self.W_target)
        comp_out = ((x @ self.V) * self.mask) @ self.U
        delta_out = F.linear(x, self.weight_delta())
        return comp_out + self.delta_mask.unsqueeze(-1) * delta_out


def install_components(model, c):
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = {}
    for path in MODULES:
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path)
        wrapper = ComponentLinear(model.get_submodule(path), c)
        setattr(parent, attr, wrapper)
        wrappers[path] = wrapper
    return wrappers


# --- CI transformer (nano section D, toy-scaled, learned pos emb, no RoPE) ---

class CIBlock(nn.Module):
    def __init__(self, d, heads, hidden):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, bias=False,
                                          batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))

    def forward(self, x):
        h = F.rms_norm(x, (x.shape[-1],))
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(F.rms_norm(x, (x.shape[-1],)))


class CITransformer(nn.Module):
    """Shared bidirectional transformer: pre-weight acts -> per-subcomponent CI."""

    def __init__(self, d_in_per_module, c, d_model=128, n_blocks=2, heads=4,
                 hidden=512, n_ctx=64, alpha=0.01):
        super().__init__()
        self.module_order = sorted(d_in_per_module)
        self.alpha = alpha
        self.proj_in = nn.Linear(sum(d_in_per_module.values()), d_model)
        self.pos = nn.Parameter(torch.zeros(n_ctx, d_model))
        self.blocks = nn.ModuleList(CIBlock(d_model, heads, hidden)
                                    for _ in range(n_blocks))
        self.proj_out = nn.Linear(d_model, c * len(self.module_order))
        self.c = c

    def forward(self, acts):
        normed = [F.rms_norm(acts[n], (acts[n].shape[-1],))
                  for n in self.module_order]
        x = self.proj_in(torch.cat(normed, -1)) + self.pos[: acts[
            self.module_order[0]].shape[1]]
        for b in self.blocks:
            x = b(x)
        logits = self.proj_out(x)
        per_module = dict(zip(self.module_order,
                              logits.split([self.c] * len(self.module_order),
                                           dim=-1)))
        ci_lower = {n: lower_leaky(v, self.alpha) for n, v in per_module.items()}
        ci_upper = {n: upper_leaky(v, self.alpha) for n, v in per_module.items()}
        return ci_lower, ci_upper


# --- losses + masks (nano section E) ---

def faithfulness_loss(wrappers):
    sum_sq, numel = 0.0, 0
    for w in wrappers.values():
        d = w.weight_delta()
        sum_sq = sum_sq + d.pow(2).sum()
        numel += d.numel()
    return sum_sq / numel


def importance_minimality_loss(ci_upper, p, eps=1e-12, beta=0.5):
    total = 0.0
    for v in ci_upper.values():
        vals = (v + eps).pow(p)
        sum_c = vals.sum(dim=(0, 1))
        mean_c = sum_c / (vals.shape[0] * vals.shape[1])
        total = total + (mean_c + beta * mean_c * torch.log2(1 + sum_c)).sum()
    return total


def kl_final(pred, target):
    """KL(target || pred) at the final position (the induction prediction)."""
    log_q = F.log_softmax(pred[:, -1], -1)
    p = F.softmax(target[:, -1].detach(), -1)
    return F.kl_div(log_q, p, reduction="none").sum(-1).mean()


def set_masks(wrappers, masks, delta_masks):
    for n, w in wrappers.items():
        w.mode, w.mask, w.delta_mask = "component", masks[n], delta_masks[n]


def clear_masks(wrappers):
    for w in wrappers.values():
        w.mode, w.mask, w.delta_mask = "target", None, None


def masked_forward_kl(model, wrappers, seq, target_logits, masks, delta_masks):
    set_masks(wrappers, masks, delta_masks)
    try:
        pred = model(seq)
    finally:
        clear_masks(wrappers)
    return kl_final(pred, target_logits)


def stochastic_recon_loss(model, wrappers, seq, target_logits, ci_lower):
    masks = {n: ci + (1 - ci) * torch.rand_like(ci)
             for n, ci in ci_lower.items()}
    dmasks = {n: torch.rand(*ci.shape[:-1], device=ci.device)
              for n, ci in ci_lower.items()}
    return masked_forward_kl(model, wrappers, seq, target_logits, masks, dmasks)


class PersistentPGD:
    """Adversarial mask sources with private Adam, persisted across steps
    (nano section F, bsc scope)."""

    def __init__(self, wrappers, b, s, device, lr=0.01, betas=(0.5, 0.99),
                 eps=1e-8, inner_steps=2):
        self.lr, self.betas, self.eps, self.inner = lr, betas, eps, inner_steps
        self.sources = {n: torch.rand(b, s, w.C + 1, device=device)
                        .requires_grad_(True) for n, w in wrappers.items()}
        self.m = {n: torch.zeros_like(s.detach()) for n, s in self.sources.items()}
        self.v = {n: torch.zeros_like(s.detach()) for n, s in self.sources.items()}
        self.t = 0

    def recon_loss(self, model, wrappers, seq, target_logits, ci_lower):
        masks = {n: ci + (1 - ci) * self.sources[n][..., : ci.shape[-1]]
                 for n, ci in ci_lower.items()}
        dmasks = {n: self.sources[n][..., -1] for n in ci_lower}
        return masked_forward_kl(model, wrappers, seq, target_logits, masks,
                                 dmasks)

    def warmup(self, model, wrappers, seq, target_logits, ci_lower):
        for _ in range(self.inner):
            loss = self.recon_loss(model, wrappers, seq, target_logits,
                                   ci_lower)
            grads = torch.autograd.grad(loss, list(self.sources.values()))
            self.step(dict(zip(self.sources, grads)))

    def step(self, grads):
        self.t += 1
        b1, b2 = self.betas
        bc1, bc2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        with torch.no_grad():
            for n, src in self.sources.items():
                g = grads[n]
                self.m[n].mul_(b1).add_(g, alpha=1 - b1)
                self.v[n].mul_(b2).addcmul_(g, g, value=1 - b2)
                src.add_(self.lr * (self.m[n] / bc1)
                         / ((self.v[n] / bc2).sqrt() + self.eps))
                src.clamp_(0.0, 1.0)


def anneal_p(step, total, p_start=2.0, p_end=0.4):
    return p_start + (p_end - p_start) * min(step / total, 1.0)


def cosine_lr(step, total, start, final_frac):
    progress = min(step / max(total - 1, 1), 1.0)
    final = start * final_frac
    return final + 0.5 * (start - final) * (1 + math.cos(math.pi * progress))


# --- training + clustering ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_per_module", type=int, default=100)
    ap.add_argument("--n_components", type=int, default=100,
                    help="model-spanning components after CI clustering")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--main_lr", type=float, default=3e-4)
    ap.add_argument("--coeff_faith", type=float, default=1e7)
    ap.add_argument("--coeff_imp", type=float, default=2e-4)
    ap.add_argument("--coeff_stoch", type=float, default=0.5)
    ap.add_argument("--coeff_ppgd", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path,
                    default=Path(__file__).parent / "out/induction_model.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    dev = args.device
    out_dir = args.out or (Path(__file__).parent
                           / f"out/vpd_C{args.n_components}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model = InductionModel().to(dev)
    model.load_state_dict(torch.load(args.ckpt)["state_dict"])
    model.eval()
    wrappers = install_components(model, args.c_per_module)
    model.to(dev)   # move freshly created V/U wrapper params to the device
    ci_fn = CITransformer({n: 16 for n in MODULES}, args.c_per_module).to(dev)
    comp_params = [p for w in wrappers.values() for p in (w.V, w.U)]
    print(f"{len(wrappers)} matrices x C={args.c_per_module} subcomponents; "
          f"CI fn {sum(p.numel() for p in ci_fn.parameters()):,} params")

    warm = torch.optim.AdamW(comp_params, lr=1e-3, weight_decay=0.0)
    for _ in range(400):
        warm.zero_grad()
        faithfulness_loss(wrappers).backward()
        warm.step()
    print(f"faithfulness warmup done: {faithfulness_loss(wrappers):.3e}")

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    ppgd = PersistentPGD(wrappers, args.batch, 64, dev)
    opt = torch.optim.AdamW(comp_params + list(ci_fn.parameters()),
                            lr=args.main_lr, weight_decay=0.0)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.main_lr, 0.1)
        seq, _, _ = gen_batch(args.batch, dev, gen)

        clear_masks(wrappers)
        target_logits = model(seq)
        acts = {n: w.last_input for n, w in wrappers.items()}
        ci_lower, ci_upper = ci_fn(acts)

        ppgd.warmup(model, wrappers, seq, target_logits, ci_lower)
        loss_faith = faithfulness_loss(wrappers)
        loss_imp = importance_minimality_loss(
            ci_upper, anneal_p(step, args.steps))
        loss_stoch = stochastic_recon_loss(model, wrappers, seq,
                                           target_logits, ci_lower)
        loss_ppgd = ppgd.recon_loss(model, wrappers, seq, target_logits,
                                    ci_lower)
        total = (args.coeff_faith * loss_faith + args.coeff_imp * loss_imp
                 + args.coeff_stoch * loss_stoch + args.coeff_ppgd * loss_ppgd)

        ppgd_grads = torch.autograd.grad(loss_ppgd,
                                         list(ppgd.sources.values()),
                                         retain_graph=True)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(comp_params, args.grad_clip)
        opt.step()
        ppgd.step(dict(zip(ppgd.sources, ppgd_grads)))

        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                l0 = torch.stack([(ci > 0.5).float().sum(-1).mean()
                                  for ci in ci_lower.values()]).sum()
            print(f"step {step:6d} faith {loss_faith:.3e} imp {loss_imp:.3f} "
                  f"stoch {loss_stoch:.4f} ppgd {loss_ppgd:.4f} "
                  f"L0(ci>.5) {l0:.1f}", flush=True)

    # -- CI profiles on held-out data -> cluster subcomponents ---------------
    with torch.no_grad():
        profiles = []
        for i in range(8):
            seq, _, _ = gen_batch(512, dev, gen)
            clear_masks(wrappers)
            model(seq)
            ci_lower, _ = ci_fn({n: w.last_input for n, w in wrappers.items()})
            profiles.append(torch.cat(
                [ci_lower[n].flatten(0, 1).T for n in sorted(MODULES)]))
        raw = torch.cat(profiles, dim=1)              # [8*C, positions]
        norms = raw.norm(dim=1, keepdim=True)
        alive = norms.squeeze(1) > 1e-3
        print(f"subcomponents alive (CI-profile norm > 1e-3): "
              f"{int(alive.sum())}/{raw.shape[0]}")
        # dead rows stay zero instead of being blown up by normalization,
        # so they cluster together at the zero centroid
        prof = torch.where(alive[:, None], raw / norms.clamp_min(1e-12),
                           torch.zeros_like(raw))
        lab = kmeans(prof, args.n_components, iters=25, seed=args.seed)

        # -- assemble model-spanning components -----------------------------
        comps, order = {}, sorted(MODULES)
        total_ci = torch.zeros(args.n_components, device=dev)
        total_ci.index_add_(0, lab, raw.sum(1))
        dump_cluster = int(total_ci.argmin())
        for mi, path in enumerate(order):
            w = wrappers[path]
            sub = torch.einsum("co,ic->coi", w.U, w.V)  # [C, d_out, d_in]
            lab_m = lab[mi * w.C:(mi + 1) * w.C]
            dense = torch.zeros(args.n_components, *w.W_target.shape,
                                device=dev)
            dense.index_add_(0, lab_m, sub)
            dense[dump_cluster] += w.weight_delta()   # exact additivity
            comps[path + ".weight"] = dense
            err = (dense.sum(0) - w.W_target).abs().max().item()
            assert err < 1e-4, (path, err)
        print(f"residual folded into cluster {dump_cluster} "
              f"(faith err {faithfulness_loss(wrappers):.3e})")

    torch.save({"components": {n: c.cpu() for n, c in comps.items()},
                "labels": lab.cpu(),
                "config": {k: str(v) for k, v in vars(args).items()}},
               out_dir / "components.pt")
    torch.save({"wrappers": {n: {"V": w.V.detach().cpu(),
                                 "U": w.U.detach().cpu()}
                             for n, w in wrappers.items()},
                "ci_fn": ci_fn.state_dict()}, out_dir / "vpd_state.pt")
    print(f"saved {out_dir}/components.pt and vpd_state.pt")


if __name__ == "__main__":
    main()
