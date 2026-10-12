"""Projected Lp concentration with one unconditional projected Adam update per step."""
from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import asdict

import parauto_core as core
import parauto_structured as structured
import parauto_tracking as tracking
import torch


@dataclass
class TrainConfig:
    steps: int = 50000
    batch: int = 2048
    lr: float = 1e-3
    lr_min: float = 1e-5
    seed: int = 0
    log_interval: int = 250
    tau: float = 0.001             # per-piece output RMS, not normalized loss units
    p_end: float = 0.5
    lp_epsilon: float = 1e-6
    normalization_events: int = 256
    precision: str = "fp32"
    projection_precision: str = "fp64"
    grad_clip: float | None = None    # total grad norm over all factors, before Adam (VPD clips components at 0.01)
    joint_fractions: tuple = ()       # global top-k keep fractions for the signed matrix-output residual loss
    joint_within: bool = False        # square the removed sum per matrix (physical cancellation only), not across matrices
    piece_rank: int = 1               # columns per piece: each piece is one matrix U_c R_c^T of rank <= piece_rank
    rank_coeff: float = 0.            # weight of the activity-weighted column-size penalty that lowers piece rank
    ownership_ratio: int = 0          # >0: components learn to own rank-one pieces; N // ratio components per matrix
    rank_price: float = 0.            # component cost per token: s^p * (pieces owned)^rank_price
    ownership_lr_scale: float = 10.   # ownership logits use this multiple of the piece learning rate
    ownership_init_prob: float = 0.5  # initial share of each piece owned by its randomly assigned component
    ownership_hard: bool = False      # straight-through: forward uses argmax ownership (as at evaluation), backward the softmax
    count_coeff: float = 0.           # additive soft activity count; zero preserves the existing objective
    count_tau: float = 0.001          # fixed score scale for s/(s+tau), separate from diagnostic tau
    density_beta: float = 0.          # VPD's ImportanceMinimalityLoss beta: price each piece by its batch usage
    local_coeff: float = 0.           # weight of the layer-local sufficiency term (see local_residuals)
    local_fractions: tuple = ()       # keep fractions the local term is trained at; one is drawn per step
    local_metric: str = "raw"         # "raw": plain squared error; "avg": caller-supplied downstream metric factor


def project(weight, pieces, cfg):
    if cfg.projection_precision == "fp64":
        return core.projected_pieces(weight, pieces)
    if cfg.projection_precision != "fp32":
        raise ValueError("Projection must use FP32 or FP64")
    return core.projected_pieces(weight, pieces, dtype=torch.float32)


@torch.no_grad()
def project_(weight, pieces, cfg):
    if cfg.projection_precision == "fp64":
        return core.project_piece_sum_(weight, pieces)
    projected = project(weight, pieces, cfg)
    pieces.read.copy_(projected.read); pieces.write.copy_(projected.write)
    if not (torch.isfinite(pieces.read).all() and torch.isfinite(pieces.write).all()):
        raise RuntimeError("Non-finite exact-sum projection")


def lp_concentration(scores, p, epsilon):
    """Zero at zero, finite derivative there, approaching |a|^p away from epsilon."""
    return ((scores.abs() + epsilon).pow(p) - epsilon ** p).sum(-1)


def concentration(scores, cfg, p):
    return lp_concentration(scores, p, cfg.lp_epsilon)


def minimality(scores, cfg, p, scale=None):
    """Per-token Lp concentration, optionally priced by how much each piece is used across the batch.

    With density_beta > 0 this takes the shape of VPD's ImportanceMinimalityLoss (its `beta` term,
    0.5 in the published run):

        sum_c mean_t a_c^p * (1 + beta * log2(1 + sum_t a_c^p))

    so a piece that fires on many tokens pays MORE per use than one that fires rarely, which pushes
    towards many rarely-active pieces rather than a few reused ones. The batch total is written as
    mean_t * cfg.batch so the fixed 256-position normalization set prices usage in the same units as
    a training step. Scores are divided by a fixed per-matrix scale first, because VPD's causal
    importances live in [0, 1] while our attribution magnitudes carry each matrix's own scale.
    With density_beta = 0 this is exactly the previous objective.
    """
    if not cfg.density_beta:
        return concentration(scores, cfg, p).mean()
    values = scores.abs() if scale is None else scores.abs() / scale
    power = (values + cfg.lp_epsilon).pow(p)
    mean = power.mean(0) - cfg.lp_epsilon ** p
    total = power.mean(0) * (cfg.batch if cfg.batch else scores.shape[0])
    return (mean * (1 + cfg.density_beta * torch.log2(1 + total))).sum()


def soft_activity_count(scores, tau):
    """Concave in magnitude; zero at zero and approaching one per active piece."""
    size = scores.abs()
    return (size / (size + tau)).sum(-1)


def slice_entry(entry, index):
    if entry.get("structured"):
        return structured.slice_entry(entry, index)
    return {key: (value if key == "output_basis" else value[index])
            for key, value in entry.items() if key != "side"}


def scores_of(entry, pieces, rank=1):
    if rank > 1:
        return structured.grouped_scores(entry, pieces, rank)
    if entry.get("structured"):
        return structured.scores(entry, pieces)
    return core.free_codes(entry["h"], entry["cot"], pieces, entry["output_basis"]) / math.sqrt(entry["cot"].shape[1])


def training_scores(entry, pieces, cfg):
    """Autocast score matmuls only; projection and Adam retain precision."""
    if cfg.precision == "fp32":
        return scores_of(entry, pieces, cfg.piece_rank)
    if cfg.precision not in ("fp16", "bf16") or not entry.get("structured"):
        raise ValueError("Mixed precision training requires the structured score backend")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg.precision]
    with torch.autocast(device_type=pieces.read.device.type, dtype=dtype):
        if entry.get('kind') == 'scalar_ig':
            # Recompute only the score matmuls during backward, keeping the
            # five-point frozen cache but avoiding five sets of saved products.
            from torch.utils.checkpoint import checkpoint
            return checkpoint(lambda read, write: structured.integrated_scalar_scores(entry, read, write),
                              pieces.read, pieces.write, use_reentrant=False)
        return scores_of(entry, pieces, cfg.piece_rank)


def backward(loss, scaler):
    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()


def optimizer_step(optimizer, scaler):
    if not scaler.is_enabled():
        optimizer.step()
        return False
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() < old_scale


def net_effect_rms(entry, weight):
    if entry.get("structured"):
        return entry["net_rms"]
    value = entry["h"] @ weight.T
    if entry["output_basis"] is not None:
        value = value @ entry["output_basis"]
    return torch.einsum("npd,nd->np", entry["cot"], value).square().mean(-1).sqrt()


def joint_parts(entry, pieces, cfg):
    """(read [n,C], sens [n,K,C], RMS score [n,C]) from the multi-probe vjp cache, computed once."""
    if entry.get('kind') != 'vjp': raise ValueError('joint residual needs the multi-probe vjp cache')
    with torch.autocast(device_type=pieces.read.device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
        read, sens = structured.signed_parts(entry, pieces)
    score = read.abs() * torch.linalg.vector_norm(sens.float(), dim=1) / math.sqrt(entry['output_count'])
    return read, sens, score


def joint_residuals(parts, cfg):
    """Signed first-order KL proxy of removing everything outside the global top-k, per keep fraction.

    z[c,k] = (r_c.h)(g_k.u_c) is piece c's signed effect on sampled direction k. The global top-k
    per token (by RMS score across all matrices, as KLKeep ranks) is kept; the loss is
    mean_{n,k} (sum_{c removed} z[c,k])^2. This includes same-source-position
    cross terms, but omits interactions between source positions. It is NOT
    twice the full-sequence quadratic KL: that requires summing signed effects
    over source positions within each sequence before squaring, then dividing
    by the number of prediction positions. Keep this historical objective
    unchanged so existing experiments remain reproducible.
    parts: per-matrix (read, sens, score). Selection is detached; gradients flow through z.
    """
    combined = torch.cat([score.detach() for _, _, score in parts], -1)
    counts = [score.shape[-1] for _, _, score in parts]
    values = []
    for fraction in cfg.joint_fractions:
        keep = core.topk_mask(combined, max(1, round(fraction * combined.shape[-1])))
        removed, within = None, 0.
        for (read, sens, _), mask in zip(parts, keep.split(counts, -1)):
            part = torch.einsum('nc,nkc->nk', (read * (~mask).to(read.dtype)).to(sens.dtype), sens).float()
            if cfg.joint_within:
                within = within + part.square().mean()     # removal is exactly linear at one matrix output
            else:
                removed = part if removed is None else removed + part
        values.append(within if cfg.joint_within else removed.square().mean())
    return torch.stack(values)


LOCAL_NONLINEARITIES = {"linear": None, "relu": torch.relu,
                        "gelu_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh")}


def local_error(entry, pieces, weight, mask, spec, cfg):
    """Mean squared layer-local error of removing every piece outside the kept mask, at one matrix.

    off = sum_{c not kept} (r_c.h) u_c is the off set's AGGREGATE write at this matrix's output, so the
    sparse matrix output on the intact input is y - off, with y = W h exact by the sum constraint. The
    error is taken after this layer's own nonlinearity phi:  e = phi(y - off) - phi(y). Only the aggregate
    has to vanish, so pieces in the off set may cancel one another; this is not the per-piece write norm.
    spec: {"nonlinearity": key of LOCAL_NONLINEARITIES, "metric": [d_out, m] factor L or None}; with L the
    error is measured as ||L^T e||^2, i.e. in the metric M = L L^T of what downstream ever reads on average.
    """
    device = pieces.read.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"):
        h = entry["h"]
        if not torch.is_autocast_enabled(device.type): h = h.to(pieces.read.dtype)
        off = ((h @ pieces.read) * (~mask).to(pieces.read.dtype)) @ pieces.write.T
        with torch.no_grad():
            y = h @ weight.to(h.dtype).T
    phi = LOCAL_NONLINEARITIES[spec["nonlinearity"]]
    promote = lambda t: t if t.dtype == torch.float64 else t.float()   # FP64 tests stay exact
    if phi is None:
        error = promote(off)                      # sign is irrelevant: the term is squared
    else:
        y = promote(y)
        error = phi(y - promote(off)) - phi(y)
    metric = spec.get("metric")
    if metric is not None:
        error = error @ metric.to(error.dtype)
    return error.square().sum(-1).mean()


def local_residuals(parts, cfg, fractions):
    """[fractions, matrices] layer-local error when only the global top-q pieces are kept.

    The support is the same global ranking KLKeep uses, detached, so gradients flow through the writes
    only. If every matrix reproduces its own output under the support, the sparse model's activations
    equal the intact model's by induction over layers, which is what makes an intact-point sensor valid.
    parts: (entry, pieces, weight, spec, scores) per matrix.
    """
    combined = torch.cat([score.detach() for *_, score in parts], -1)
    counts = [score.shape[-1] for *_, score in parts]
    rows = []
    for fraction in fractions:
        keep = core.topk_mask(combined, max(1, round(fraction * combined.shape[-1])))
        rows.append(torch.stack([local_error(entry, pieces, weight, mask, spec, cfg)
                                 for (entry, pieces, weight, spec, _), mask
                                 in zip(parts, keep.split(counts, -1), strict=True)]))
    return torch.stack(rows)


def rank_penalty(pieces, scores, cfg, p):
    """Activity-weighted Lp of column sizes ||u_j|| ||r_j|| inside each rank-k piece.

    Driving a column's size to zero lowers that piece's rank. At p = 1, the minimum of sum_j ||u_j|| ||r_j||
    over rotations inside the piece is the piece's nuclear norm. Weights: each piece's batch-mean score over
    the matrix mean (detached), so rarely used pieces pay little for rank.
    """
    size = pieces.write.float().norm(dim=0) * pieces.read.float().norm(dim=0)             # [C*k]
    per_piece = lp_concentration(size.unflatten(-1, (-1, cfg.piece_rank)), p, cfg.lp_epsilon)  # [C]
    activity = scores.detach().float().mean(0)
    return (activity / activity.mean().clamp_min(1e-30) * per_piece).mean()


def initial_ownership(count, cfg, device):
    """Logits [N, M]: each piece gives ownership_init_prob to one random component (N/M pieces each), the rest uniform."""
    components = count // cfg.ownership_ratio
    generator = torch.Generator(device='cpu').manual_seed(cfg.seed + 303)
    owner = (torch.arange(count) % components)[torch.randperm(count, generator=generator)]
    logits = torch.zeros(count, components)
    prob = cfg.ownership_init_prob
    logits[torch.arange(count), owner] = math.log(prob * (components - 1) / (1 - prob))
    return logits.to(device)


def ownership_matrix(logits):
    return torch.softmax(logits.float(), dim=-1)


def ownership_forward(logits, cfg):
    """Soft shares, or with ownership_hard a one-hot argmax in the forward pass and the softmax gradient backward."""
    soft = ownership_matrix(logits)
    if not cfg.ownership_hard:
        return soft
    hard = torch.nn.functional.one_hot(soft.argmax(-1), soft.shape[-1]).to(soft.dtype)
    return hard + soft - soft.detach()


def component_cost(scores, rank, cfg, p):
    """Per token: sum_m ((s_m + eps)^p - eps^p) * rank_m^rank_price, rank_m = pieces owned (soft count)."""
    price = rank.float().clamp_min(1e-2).pow(cfg.rank_price)
    return (((scores.abs() + cfg.lp_epsilon).pow(p) - cfg.lp_epsilon ** p) * price).sum(-1)


def owned_training_scores(entry, pieces, ownership, cfg):
    if cfg.precision == "fp32":
        return structured.soft_component_scores(entry, pieces, ownership)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[cfg.precision]
    with torch.autocast(device_type=pieces.read.device.type, dtype=dtype):
        return structured.soft_component_scores(entry, pieces, ownership)


def clone_pieces(p):
    return core.FreePieces(p.read.detach().clone(), p.write.detach().clone())


def prepare_initial(weight, components, cfg, log=print):
    """Initialize free factors and project their sum exactly to the matrix."""
    pieces = core.initialize_pieces(weight, components, cfg.seed)
    info = {"initialization": "random_free_factors_then_exact_sum_projection",
            "seed": cfg.seed, "matrix_sum_relative_error": core.reconstruction_error(weight, pieces)}
    log(info)
    return pieces, info


def fit_matrix(entry, weight, initial, cfg, log=print):
    pieces, histories = fit([entry], [weight], [initial], cfg,
        log=lambda row: log({k: v for k, v in row.items() if k not in ('matrix_index', 'total_loss')}))
    return pieces[0], histories[0]


def fit(entries, weights, initial, cfg, log=print, *, resume=None,
        checkpoint=None, checkpoint_interval=1000, stop_after=None,
        batch_provider=None, step_callback=None, allow_partial_batch=False,
        sync_initial=None, after_backward=None, after_step=None, local=None):
    """Sum independent matrix losses, with one backward and Adam step per batch.

    Each matrix retains its normalization, random stream and exact-sum projection.
    Summing (not averaging) preserves the separate fitter's parameter gradients.
    Adam updates are followed directly by projection, with no trial moves or rollback.
    Data-parallel hooks (all optional, no effect when None): sync_initial(tensors) makes normalizers
    identical across replicas; after_backward(parameters) averages gradients before clipping and Adam;
    after_step(next_step, pieces) runs after the projection.
    An optional batch_provider(step) supplies one aligned fresh batch for all
    matrices, used once without resampling. step_callback(next_step) may request
    a clean stop; its final state is checkpointed using the same resume format.
    """
    if not entries or not len(entries) == len(weights) == len(initial):
        raise ValueError("Expected matching nonempty entries, weights and initial factors")
    if not 0 < cfg.p_end < 1 or cfg.tau <= 0 or cfg.lp_epsilon <= 0:
        raise ValueError("Invalid objective configuration")
    if cfg.piece_rank < 1 or any(item.read.shape[1] % cfg.piece_rank for item in initial):
        raise ValueError("Column counts must be multiples of piece_rank")
    if cfg.piece_rank > 1 and cfg.joint_fractions:
        raise ValueError("Joint residual loss is not implemented for rank-k pieces")
    if cfg.rank_coeff and cfg.piece_rank == 1:
        raise ValueError("rank_coeff needs piece_rank > 1")
    if cfg.density_beta < 0 or not math.isfinite(cfg.density_beta):
        raise ValueError("density_beta must be finite and nonnegative")
    if cfg.density_beta and (cfg.count_coeff or cfg.ownership_ratio or cfg.piece_rank != 1):
        raise ValueError("The usage-density term supports rank-one pieces without ownership or the activity count")
    if cfg.local_coeff:
        if not cfg.local_fractions or local is None or len(local) != len(entries):
            raise ValueError("The layer-local term needs local_fractions and one local spec per matrix")
        if cfg.ownership_ratio or cfg.piece_rank != 1:
            raise ValueError("The layer-local term supports rank-one pieces without ownership")
        if any(spec["nonlinearity"] not in LOCAL_NONLINEARITIES for spec in local):
            raise ValueError("Unknown local nonlinearity")
        if cfg.local_metric not in ("raw", "avg"):
            raise ValueError("local_metric must be raw or avg")
        if (cfg.local_metric == "avg") != all(spec.get("metric") is not None for spec in local):
            raise ValueError("local_metric=avg needs a metric factor for every matrix; raw needs none")
    elif local is not None:
        raise ValueError("Local specs were supplied without local_coeff")
    owning = cfg.ownership_ratio > 0
    if not math.isfinite(cfg.count_coeff) or cfg.count_coeff < 0 or not math.isfinite(cfg.count_tau) or cfg.count_tau <= 0:
        raise ValueError('Activity count requires nonnegative coefficient and positive finite tau')
    if cfg.count_coeff and (owning or cfg.piece_rank != 1 or cfg.joint_fractions or cfg.rank_coeff):
        raise ValueError('Activity count pilot supports ordinary rank-one concentration only')
    if owning and (cfg.piece_rank > 1 or cfg.joint_fractions or any(item.read.shape[1] % cfg.ownership_ratio for item in initial)):
        raise ValueError("Ownership needs rank-one pieces, no joint loss, and counts divisible by ownership_ratio")
    pieces = [clone_pieces(item) for item in initial]
    parameters = [tensor.requires_grad_() for item in pieces for tensor in (item.read, item.write)]
    ownership = [initial_ownership(item.read.shape[1], cfg, weights[0].device).requires_grad_() for item in initial] if owning else []
    groups = [{"params": parameters, "lr_scale": 1.}] + ([{"params": ownership, "lr_scale": cfg.ownership_lr_scale}] if owning else [])
    parameters = parameters + ownership
    optimizer = torch.optim.Adam(groups, lr=cfg.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
    overflow_skips = 0
    # Separate seeded generators reproduce every matrix's existing batch
    # sequence, including when entries have different numbers of examples.
    batches = [torch.Generator(device=w.device).manual_seed(cfg.seed + 202) for w in weights]
    normalization = [slice_entry(entry, slice(0, min(cfg.normalization_events, len(entry["h"])))) for entry in entries]
    if owning:
        with torch.no_grad():
            initial_mix = [ownership_forward(o, cfg) for o in ownership]
            initial_scores = [owned_training_scores(reference, item, a, cfg).float() for reference, item, a in zip(normalization, initial, initial_mix, strict=True)]
            initial_owner_rank = [a.sum(0) for a in initial_mix]
    else:
        initial_scores = [scores_of(reference, item, cfg.piece_rank).detach() for reference, item in zip(normalization, initial, strict=True)]
        initial_owner_rank = None
    initial_rank = ([rank_penalty(item, score, cfg, 1.0).detach() for item, score in zip(initial, initial_scores, strict=True)]
                    if cfg.rank_coeff else None)
    initial_joint = (joint_residuals([joint_parts(r, item, cfg) for r, item in zip(normalization, initial, strict=True)], cfg).detach()
                     if cfg.joint_fractions else None)
    local_choice = torch.Generator(device="cpu").manual_seed(cfg.seed + 404)
    if cfg.local_coeff:
        with torch.no_grad():
            initial_local = local_residuals(list(zip(normalization, initial, weights, local, initial_scores, strict=True)),
                                            cfg, cfg.local_fractions).detach()
    else:
        initial_local = None
    if sync_initial is not None:
        sync_initial(initial_scores + ([initial_joint] if initial_joint is not None else [])
                     + ([initial_local] if initial_local is not None else []) + (initial_rank or []))
    histories = [[] for _ in entries]
    start = 0
    if resume is not None:
        if {**asdict(TrainConfig()), **resume['config']} != asdict(cfg): raise ValueError('Resume training configuration mismatch')
        for item, saved in zip(pieces, resume['pieces'], strict=True):
            with torch.no_grad():
                item.read.copy_(saved['read']); item.write.copy_(saved['write'])
        optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
        for gen, state in zip(batches, resume['batches'], strict=True): gen.set_state(state.cpu())
        initial_scores = [v.to(weights[0].device) for v in resume['initial_scores']]
        if cfg.joint_fractions: initial_joint = resume['initial_joint'].to(weights[0].device)
        if cfg.rank_coeff: initial_rank = [v.to(weights[0].device) for v in resume['initial_rank']]
        if cfg.local_coeff:
            initial_local = resume['initial_local'].to(weights[0].device)
            local_choice.set_state(resume['local_choice'].cpu())
        if owning:
            with torch.no_grad():
                for o, saved in zip(ownership, resume['ownership'], strict=True): o.copy_(saved)
            initial_owner_rank = [v.to(weights[0].device) for v in resume['initial_owner_rank']]
        histories = resume['histories']
        start, overflow_skips = resume['next_step'], resume['overflow_skips']
    # Fixed per-matrix score scale, so the usage-density log sees a dimensionless activity.
    density_scales = ([v.abs().mean().clamp_min(1e-30).detach() for v in initial_scores]
                      if cfg.density_beta else None)
    # One GLOBAL fixed normalizer: each component has the same count price.
    # Scale initial total count to len(entries), matching the initial summed Lp loss.
    count_denominator = (torch.stack([soft_activity_count(s, cfg.count_tau).mean()
                         for s in initial_scores]).sum().detach().clamp_min(1e-8) / len(entries)
                         if cfg.count_coeff else None)
    end = cfg.steps if stop_after is None else min(cfg.steps, start + stop_after)
    for step in range(start, end):
        for group in optimizer.param_groups:
            group["lr"] = core.cosine_lr(step, cfg) * group.get("lr_scale", 1.)
        p = 1 + (cfg.p_end - 1) * min(1.0, step / max(1, cfg.steps * 0.8))
        optimizer.zero_grad(set_to_none=True)
        step_entries = entries if batch_provider is None else batch_provider(step)
        if len(step_entries) != len(entries):
            raise ValueError('Batch provider matrix count mismatch')
        losses, diagnostics, parts, local_parts = [], [], [], []
        for i, (entry, weight, item) in enumerate(zip(step_entries, weights, pieces, strict=True)):
            if batch_provider is None:
                index = torch.randint(len(entry["h"]), (min(cfg.batch, len(entry["h"])),),
                                      device=weight.device, generator=batches[i])
                batch = slice_entry(entry, index)
            else:
                if len(entry['h']) != cfg.batch and not (allow_partial_batch and 0 < len(entry['h']) < cfg.batch):
                    raise ValueError('Fresh batch must contain exactly cfg.batch positions')
                batch = entry
            feasible = project(weight, item, cfg)
            if owning:
                mix = ownership_forward(ownership[i], cfg)
                scores = owned_training_scores(batch, feasible, mix, cfg)
                denominator = component_cost(initial_scores[i], initial_owner_rank[i], cfg, p).mean().detach().clamp_min(1e-8)
                sparse = component_cost(scores, mix.sum(0), cfg, p).mean() / denominator
                losses.append(sparse)
                diagnostics.append((sparse.detach(), scores.detach(), None))
                continue
            if cfg.joint_fractions:
                read, sens, scores = joint_parts(batch, feasible, cfg)
                parts.append((read, sens, scores))
            else:
                scores = training_scores(batch, feasible, cfg)
            if cfg.local_coeff:
                local_parts.append((batch, feasible, weight, local[i], scores))
            scale = density_scales[i] if density_scales is not None else None
            denominator = minimality(initial_scores[i], cfg, p, scale).detach().clamp_min(1e-8)
            sparse = minimality(scores, cfg, p, scale) / denominator
            rank_term = None
            if cfg.rank_coeff:
                # Normalized by the initial pieces' penalty at the same p, so both terms start near 1.
                start = rank_penalty(initial[i], initial_scores[i], cfg, p).detach().clamp_min(1e-12) if p != 1.0 else initial_rank[i].clamp_min(1e-12)
                rank_term = rank_penalty(feasible, scores, cfg, p) / start
            losses.append(sparse if rank_term is None else sparse + cfg.rank_coeff * rank_term)
            if cfg.count_coeff:
                losses[-1] = losses[-1] + cfg.count_coeff * soft_activity_count(scores, cfg.count_tau).mean() / count_denominator
            diagnostics.append((sparse.detach(), scores.detach(), None if rank_term is None else rank_term.detach()))
        loss = torch.stack(losses).sum()
        if cfg.joint_fractions:
            joint = joint_residuals(parts, cfg) / initial_joint.clamp_min(1e-12)
            loss = loss + joint.sum()
            tracking_joint = [float(v) for v in joint.detach()]
            del parts
        if cfg.local_coeff:
            # One keep fraction per step, drawn from the range KLKeep will test, so the off set
            # has to be removable along the whole prefix of the curve rather than at one budget.
            choice = int(torch.randint(len(cfg.local_fractions), (1,), generator=local_choice))
            local_term = local_residuals(local_parts, cfg, (cfg.local_fractions[choice],))[0] / initial_local[choice].clamp_min(1e-12)
            loss = loss + cfg.local_coeff * local_term.sum()
            tracking_local = (cfg.local_fractions[choice], [float(v) for v in local_term.detach()])
            del local_parts, local_term
        backward(loss, scaler)
        if after_backward is not None:
            after_backward(parameters)
        if cfg.grad_clip is not None:
            if scaler.is_enabled(): scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
        overflow_skips += int(optimizer_step(optimizer, scaler))
        for i, (weight, item) in enumerate(zip(weights, pieces, strict=True)):
            project_(weight, item, cfg)
            if step % cfg.log_interval == 0 or step == cfg.steps - 1:
                error = core.reconstruction_error(weight, item)
                if error > 1e-5:
                    raise RuntimeError(f"Matrix sum drift: {error}")
                sparse, scores, rank_term = diagnostics[i]
                row = {"step": step, "p": p, "loss": float(losses[i].detach()),
                       "concentration_normalized": float(sparse),
                       "optimizer_updates": step + 1 - overflow_skips,
                       "reconstruction_relative_error": error,
                       "batch_threshold_count": float((scores > cfg.tau).sum(-1).float().mean())}
                histories[i].append(row)
                if cfg.count_coeff:
                    row['soft_activity_count'] = float(soft_activity_count(scores, cfg.count_tau).mean())
                    row['count_normalized'] = row['soft_activity_count'] / float(count_denominator)
                if scaler.is_enabled():
                    row.update(gradient_scale=scaler.get_scale(), overflow_skipped_updates=overflow_skips)
                if cfg.joint_fractions: row["joint_normalized"] = tracking_joint
                if cfg.local_coeff:
                    row["local_fraction"], row["local_normalized"] = tracking_local[0], tracking_local[1][i]
                if owning:
                    with torch.no_grad():
                        mix = ownership_matrix(ownership[i])
                        sizes = torch.bincount(mix.argmax(-1), minlength=mix.shape[1])
                        row.update(components_used=int((sizes > 0).sum()), component_size_max=int(sizes.max()),
                                   component_size_mean_used=float(sizes[sizes > 0].float().mean()),
                                   ownership_max_mean=float(mix.max(-1).values.mean()))
                if cfg.piece_rank > 1:
                    sv = core.piece_singular_values(item, cfg.piece_rank)
                    row["effective_rank_mean"] = float((sv > 1e-2 * sv[:, :1].clamp_min(1e-30)).sum(-1).double().mean())
                    row["participation_rank_mean"] = float((sv.sum(-1).square() / sv.square().sum(-1).clamp_min(1e-30)).mean())
                    if rank_term is not None: row["rank_normalized"] = float(rank_term)
                event = {"matrix_index": i, "total_loss": float(loss.detach()), **row}
                tracking.log_train(event)
                log(event)
        # Release token-sized intermediates before the next cache generation.
        del step_entries, batch, entry, scores, diagnostics, losses, loss, feasible, sparse
        if after_step is not None:
            after_step(step + 1, pieces)
        stop_requested = bool(step_callback(step + 1)) if step_callback is not None else False
        if checkpoint is not None and ((step + 1) % checkpoint_interval == 0 or step == end - 1 or stop_requested):
            checkpoint({'config': asdict(cfg), 'optimizer_version': 'projected_adam_no_rejection_v1', 'next_step': step + 1,
                        'pieces': [{'read': p.read.detach(), 'write': p.write.detach()} for p in pieces],
                        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                        'batches': [g.get_state() for g in batches],
                        'initial_scores': initial_scores, 'initial_joint': initial_joint, 'initial_rank': initial_rank, 'histories': histories,
                        'initial_local': initial_local, 'local_choice': local_choice.get_state(),
                        'ownership': [o.detach() for o in ownership] if owning else None, 'initial_owner_rank': initial_owner_rank,
                        'overflow_skips': overflow_skips})
        if stop_requested:
            break
    return [clone_pieces(item) for item in pieces], histories
