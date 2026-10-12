"""Binary support selection and exact-sum factor training; no learned selector.

All losses are mean over prediction positions within each sequence. Gate
derivatives are taken from the SUM of those per-sequence losses. Thus a gate
gradient, a sketched curvature, and rate/T use identical units for any batch.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[2]))
import parauto_core as core

KINDS = ('attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj',
         'mlp.c_fc', 'mlp.down_proj')
VARIANTS = ('error_gradient', 'replacement_cost', 'restoration_path')


@dataclass(frozen=True)
class Config:
    steps: int = 5000
    contexts: int = 64
    microbatch: int = 4
    lr: float = 1e-3
    lr_min: float = 1e-5
    seed: int = 0
    start_budget: int = 8192
    end_budget: int = 256
    anneal_steps: int = 4000
    final_budgets: tuple = (128, 192, 256, 512)
    rate: float = 1e-3  # nats per retained component per prediction position
    probes: int = 4
    shortlist: int = 2
    selection_rounds: int = 1
    max_moves: int = 128  # at most one candidate per token, 128 tokens/sequence
    precision: str = 'bf16'
    grad_clip: float = 1.


def names(model):
    return tuple(f'h.{l}.{kind}' for l in range(len(model.h)) for kind in KINDS)


def autocast(model, cfg):
    device = next(model.parameters()).device.type
    return torch.autocast(device, dtype=torch.bfloat16) if cfg.precision == 'bf16' and device == 'cuda' else nullcontext()


def budget_at(step, total, cfg):
    if step >= cfg.anneal_steps:
        return min(total, cfg.final_budgets[(step-cfg.anneal_steps) % len(cfg.final_budgets)])
    fraction = step / max(1, cfg.anneal_steps-1)
    start, end = min(total, cfg.start_budget), min(total, cfg.end_budget)
    return max(1, round(start * (end/start)**fraction))


def generator(model, seed):
    return torch.Generator(device=next(model.parameters()).device).manual_seed(seed)


@contextmanager
def factor_forwards(model, pieces, gate, *, factor_grad=False):
    """Replace linear FORWARDS, so the unused dense linear is never computed.

    Gate derivatives require all candidate columns, including currently off
    ones. These are dense factor GEMMs, not claimed to be sparse kernels.
    Restores each instance's original forward, including on exceptions.
    """
    masks = gate.split([p.count for p in pieces], dim=-1)
    originals = []
    try:
        for name, p, mask in zip(names(model), pieces, masks, strict=True):
            layer = model.get_submodule(name)
            originals.append((layer, layer.__dict__.get('forward')))
            read, write = (p.read, p.write) if factor_grad else (p.read.detach(), p.write.detach())
            def forward(x, read=read, write=write, mask=mask, bias=layer.bias):
                result = ((x @ read) * mask) @ write.T
                return result if bias is None else result+bias
            layer.forward = forward
        yield
    finally:
        for layer, original in originals:
            if original is None:
                del layer.forward
            else:
                layer.forward = original


def run(model, ids, pieces, gate, cfg, *, factor_grad=False):
    with factor_forwards(model, pieces, gate, factor_grad=factor_grad), autocast(model, cfg):
        return model(ids)[0].float()


def kl_per_sequence(logits, teacher_logp):
    return (teacher_logp.exp() * (teacher_logp-logits.float().log_softmax(-1))).sum(-1).mean(-1)


def stable_kl_difference(plus_logits, minus_logits, teacher_logp):
    """Validation-only D(plus)-D(minus), without subtracting large FP32 KLs.

    The teacher entropy cancels exactly. Forward values are unmodified; only
    the log-softmax and reduction used to measure the change run in FP64.
    """
    delta = minus_logits.double().log_softmax(-1)-plus_logits.double().log_softmax(-1)
    return (teacher_logp.double().exp()*delta).sum(-1).mean(-1)


def objective(kl, support, rate):
    return kl + rate*support.float().sum((-1, -2))/support.shape[-2]


def top_support(score, budget):
    chosen = score.topk(min(budget, score.shape[-1]), dim=-1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(-1, chosen, True)


def dense_reference(model, ids, pieces, cfg, seed, budget, *, probe_count=1, return_scores=False):
    """One ordinary sampled-label VJP initializes every arm's support.

    Identical per-microbatch seeds across arms. The teacher is detached. No
    gold labels, GIM rule, learned CI, or validation data enters selection.
    """
    inputs, outputs, handles = {}, {}, []
    try:
        def capture(name):
            def hook(_m, args, output):
                inputs[name], outputs[name] = args[0], output
            return hook
        for name in names(model):
            handles.append(model.get_submodule(name).register_forward_hook(capture(name)))
        handles.append(model.wte.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_()))
        with torch.enable_grad(), autocast(model, cfg):
            logits = model(ids)[0].float()
            logp = logits.detach().log_softmax(-1)
            labels = torch.multinomial(logp.flatten(0, 1).exp(), probe_count, replacement=True,
                                       generator=generator(model, seed))
            energies = None
            for k in range(probe_count):
                ce = F.cross_entropy(logits.flatten(0, 1), labels[:, k], reduction='sum')
                grads = torch.autograd.grad(ce, tuple(outputs[n] for n in names(model)), retain_graph=k < probe_count-1)
                with torch.no_grad(), autocast(model, cfg):
                    values = [(g.detach() @ p.write.detach()).float().square() for p, g in zip(pieces, grads, strict=True)]
                    if energies is None:
                        energies = values
                    else:
                        for energy, value in zip(energies, values, strict=True):
                            energy.add_(value)
            with torch.no_grad(), autocast(model, cfg):
                scores = [(inputs[n].detach() @ p.read.detach()).float().abs()*(energy/probe_count).sqrt()
                          for n, p, energy in zip(names(model), pieces, energies, strict=True)]
                scores = torch.cat(scores, -1)
                support = top_support(scores, budget)
        return (logp, support, scores) if return_scores else (logp, support)
    finally:
        for h in handles:
            h.remove()


def shortlist(grad, support, width):
    width = min(width, int(support.sum(-1).min()), int((~support).sum(-1).min()))
    if width == 0:
        return None
    drop = grad.masked_fill(~support, -torch.inf).topk(width, -1).indices
    add = grad.masked_fill(support, torch.inf).topk(width, -1, largest=False).indices
    return drop, add


def fisher_seed(prob, gen):
    """Zero-mean Gaussian logit probe with covariance diag(q)-qq^T."""
    z = torch.randn(prob.shape, device=prob.device, dtype=prob.dtype, generator=gen)
    root_z = prob.sqrt()*z
    return root_z-prob*root_z.sum(-1, keepdim=True)


def candidate_tensors(grad, support, cfg, *, add_gradient=None, indices=None, effects=None):
    """Best drop, swap, or 2-out/1-in proposal per token.

    Each candidate stores up to three (component, signed gate change) pairs.
    Curvature uses the norm of the SIGNED sum, retaining cross terms.
    """
    search = indices or shortlist(grad, support, cfg.shortlist)
    if search is None:
        return None
    drop, add = search
    dg = grad.gather(-1, drop)
    ag = (grad if add_gradient is None else add_gradient).gather(-1, add)
    width = drop.shape[-1]
    locations, changes, costs = [], [], []
    tax = cfg.rate/support.shape[-2]
    def append(idx, delta, linear, sketch=None):
        if sketch is not None:
            linear = linear+.5*sketch.square().mean(-1)
        linear = linear+tax*sum(delta)
        padded_idx = idx+[idx[0]]*(3-len(idx))
        padded_delta = delta+[0.]*(3-len(delta))
        locations.append(torch.stack(padded_idx, -1))
        changes.append(torch.tensor(padded_delta, device=grad.device).expand(*grad.shape[:-1], 3))
        costs.append(linear)
    # effects: [B,T,2*width,K], already normalized for mean-token KL.
    for i in range(width):
        ed = None if effects is None else effects[..., i, :]
        append([drop[..., i]], [-1.], -dg[..., i], None if ed is None else -ed)
        for j in range(width):
            ea = None if effects is None else effects[..., width+j, :]
            append([drop[..., i], add[..., j]], [-1., 1.], -dg[..., i]+ag[..., j],
                   None if ed is None else ea-ed)
    if effects is not None and width >= 2:
        for j in range(width):
            delta_effect = effects[..., width+j, :]-effects[..., 0, :]-effects[..., 1, :]
            append([drop[..., 0], drop[..., 1], add[..., j]], [-1., -1., 1.],
                   -dg[..., 0]-dg[..., 1]+ag[..., j], delta_effect)
    costs = torch.stack(costs, -1)
    best_cost, which = costs.min(-1)
    loc = torch.stack(locations, -2).gather(-2, which[..., None, None].expand(*which.shape, 1, 3)).squeeze(-2)
    delta = torch.stack(changes, -2).gather(-2, which[..., None, None].expand(*which.shape, 1, 3)).squeeze(-2)
    return dict(cost=best_cost, indices=loc, delta=delta)


def apply_candidates(support, proposal, moves):
    """Disjoint token proposals; zero padding never overwrites a real change."""
    candidate = support.clone()
    for b in range(len(support)):
        cost = proposal['cost'][b]
        chosen = cost.topk(min(moves, len(cost)), largest=False).indices
        chosen = chosen[cost[chosen] < 0]
        for slot in range(3):
            delta = proposal['delta'][b, chosen, slot]
            valid = delta != 0
            t = chosen[valid]
            c = proposal['indices'][b, t, slot]
            candidate[b, t, c] = delta[valid] > 0
    return candidate


def proposals(model, ids, pieces, support, teacher_logp, cfg, variant, seed):
    with torch.enable_grad():
        gate = support.float().requires_grad_()
        logits = run(model, ids, pieces, gate, cfg)
        kl = kl_per_sequence(logits, teacher_logp)
        grad, = torch.autograd.grad(kl.sum(), gate, retain_graph=variant == 'replacement_cost')
        info = {}
        if variant == 'replacement_cost':
            indices = shortlist(grad, support, cfg.shortlist)
            if indices is None:
                return None, kl.detach(), info
            candidates = torch.cat(indices, -1)
            prob = logits.detach().softmax(-1)
            gen = generator(model, seed)
            sketches = []
            for k in range(cfg.probes):
                # The complete sequence shares a probe. Time is summed BEFORE
                # squaring, including interactions between source positions.
                v = fisher_seed(prob, gen)/math.sqrt(ids.shape[1])
                cot, = torch.autograd.grad((v*logits).sum(), gate, retain_graph=k < cfg.probes-1)
                sketches.append(cot.gather(-1, candidates).detach())
            effects = torch.stack(sketches, -1)
            proposal = candidate_tensors(grad, support, cfg, indices=indices, effects=effects)
            info['curvature_probes'] = cfg.probes
        elif variant == 'restoration_path':
            alpha = torch.rand((), device=ids.device, generator=generator(model, seed))
            path_gate = (support.float()+alpha*(~support).float()).requires_grad_()
            path_logits = run(model, ids, pieces, path_gate, cfg)
            path_kl = kl_per_sequence(path_logits, teacher_logp)
            path_grad, = torch.autograd.grad(path_kl.sum(), path_gate)
            # Endpoint deletion cost, signed restoration credit for additions.
            # This is a proposal heuristic, not a local Taylor prediction.
            search = shortlist(grad, support, cfg.shortlist)
            if search is not None:
                add = path_grad.masked_fill(support, torch.inf).topk(search[0].shape[-1], -1, largest=False).indices
                proposal = candidate_tensors(grad, support, cfg, add_gradient=path_grad, indices=(search[0], add))
            else:
                proposal = None
            info.update(alpha=float(alpha), path_kl=float(path_kl.detach().mean()),
                        signed_restoration_credit=float((-path_grad*(~support)).sum(-1).sum(-1).mean()))
        else:
            assert variant == 'error_gradient'
            proposal = candidate_tensors(grad, support, cfg)
    if proposal is not None:
        proposal = {k: v.detach() for k, v in proposal.items()}
    return proposal, kl.detach(), info


def select(model, ids, pieces, initial, teacher_logp, cfg, variant, seed):
    support = initial.clone()
    before = None
    metrics = {}
    for iteration in range(cfg.selection_rounds):
        proposal, base_kl, info = proposals(model, ids, pieces, support, teacher_logp, cfg, variant, seed+iteration)
        base_obj = objective(base_kl, support, cfg.rate)
        if before is None:
            before = base_obj
        best, best_kl, best_obj = support.clone(), base_kl.clone(), base_obj.clone()
        accepted = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
        if proposal is not None:
            with torch.no_grad():
                for moves in sorted({cfg.max_moves, min(32, cfg.max_moves), min(8, cfg.max_moves), 1}, reverse=True):
                    candidate = apply_candidates(support, proposal, moves)
                    if torch.equal(candidate, support):
                        break
                    candidate_kl = kl_per_sequence(run(model, ids, pieces, candidate.float(), cfg), teacher_logp)
                    candidate_obj = objective(candidate_kl, candidate, cfg.rate)
                    improve = (~accepted)&torch.isfinite(candidate_obj)&(candidate_obj < best_obj-1e-7)
                    best[improve], best_kl[improve], best_obj[improve] = candidate[improve], candidate_kl[improve], candidate_obj[improve]
                    accepted |= improve
                    if bool(accepted.all()):
                        break
        support = best
        assert bool((best_obj <= base_obj+1e-7).all()), (best_obj, base_obj)
        metrics.update(info)
        metrics.update(initial_objective=float(before.mean()), selected_objective=float(best_obj.mean()),
                       selected_kl=float(best_kl.mean()), acceptance_fraction=float(accepted.float().mean()),
                       kept_per_token=float(support.float().sum(-1).mean()),
                       changed_per_token=float((support != initial).float().sum(-1).mean()))
    return support, metrics


def project_all(model, pieces):
    for name, p in zip(names(model), pieces, strict=True):
        core.project_piece_sum_(model.get_submodule(name).weight, p, dtype=torch.float32)


def clone_pieces(initial, grad=True):
    return [core.FreePieces(p.read.detach().clone().requires_grad_(grad), p.write.detach().clone().requires_grad_(grad)) for p in initial]


def create_optimizer(pieces, cfg):
    return torch.optim.Adam([x for p in pieces for x in (p.read, p.write)], lr=cfg.lr, betas=(.9, .99))


def train_step(model, rows, pieces, optimizer, cfg, variant, step):
    optimizer.zero_grad(set_to_none=True)
    lr = cfg.lr_min+.5*(cfg.lr-cfg.lr_min)*(1+math.cos(math.pi*step/max(1, cfg.steps-1)))
    for group in optimizer.param_groups:
        group['lr'] = lr
    budget = budget_at(step, sum(p.count for p in pieces), cfg)
    totals = {}
    start = time.perf_counter()
    for first in range(0, len(rows), cfg.microbatch):
        ids = rows[first:first+cfg.microbatch, :-1]
        weight = len(ids)/len(rows)
        seed = 31000000+step*cfg.contexts+first
        logp, initial = dense_reference(model, ids, pieces, cfg, seed, budget)
        support, metrics = select(model, ids, pieces, initial, logp, cfg, variant, seed+50000000)
        logits = run(model, ids, pieces, support.float(), cfg, factor_grad=True)
        kl = kl_per_sequence(logits, logp).mean()
        assert bool(torch.isfinite(kl))
        (kl*weight).backward()
        metrics['train_kl'] = float(kl.detach())
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.)+weight*value
        del logits, kl, logp, initial, support
    params = [x for p in pieces for x in (p.read, p.write)]
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in params)
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    project_all(model, pieces)
    totals.update(lr=lr, budget=budget, grad_norm=float(norm), seconds=time.perf_counter()-start)
    return totals
