"""Exact output-effect norms for the two-layer residual toy, in eager PyTorch.

Cache small norm-preserving QR factors, not full output Jacobians. Output
writes use shared linear maps plus the downstream ReLU mask. All trainable
read/write vectors retain their original dimensions.
"""
from __future__ import annotations

import math

import torch


def slice_entry(entry, index):
    # finite_ablation keeps token-level arrays in events and whole-sequence attention in shared.
    return {**entry, "h": entry["h"][index], "net_rms": entry["net_rms"][index],
            "events": {k: v[index] for k, v in entry["events"].items()}}


def norm_input(value):
    """Accumulate squared norms and residual sums in FP32 under autocast."""
    return value.float() if value.dtype in (torch.bfloat16, torch.float16) else value


def squared_sensitivity(entry, write):
    """Return [batch,C], or [C] when the write sensitivity is shared."""
    shared, events = entry["shared"], entry["events"]
    kind = entry["kind"]
    if kind == "vjp":
        cot = events['cot']
        if not torch.is_autocast_enabled(write.device.type): cot = cot.to(write.dtype)
        effect = cot @ write
        # Keep the large [tokens, probes, components] activation in FP16.
        # Promote only the reduced norm, before the epsilon/sqrt and loss.
        return torch.linalg.vector_norm(effect, dim=-2).float().square()
    if kind == "input_qr":
        effect = events["factor"] @ write
        return norm_input(effect).square().sum(-2)
    if kind == "input_mask":
        factor = shared["factor"].unsqueeze(0) * events["mask"].unsqueeze(1)
        return norm_input(factor @ write).square().sum(-2)
    if kind == "output_shared":
        return norm_input(shared["output"] @ write).square().sum(0)
    if kind == "output_residual":
        base = shared["output"] @ write
        rank = shared["response"].shape[0]
        hidden = events["mask"].unsqueeze(-1) * (shared["read"] @ write)
        effect = norm_input(shared["response"] @ hidden) + norm_input(base[:rank])
        return effect.square().sum(-2) + norm_input(base[rank:]).square().sum(0)
    raise ValueError(kind)


def signed_parts(entry, pieces):
    """For the 'vjp' kind: read [n,C] and per-probe write sensitivity [n,K,C], so that the signed
    per-probe score is read[:,None,:]*sens and the RMS score is |read| * ||sens||_K / sqrt(K)."""
    auto = torch.is_autocast_enabled(pieces.read.device.type)
    h, cot = entry['h'], entry['events']['cot']
    if not auto: h, cot = h.to(pieces.read.dtype), cot.to(pieces.write.dtype)
    return norm_input(h @ pieces.read), cot @ pieces.write


def grouped_scores(entry, pieces, rank):
    """Score of whole rank-k pieces P_c = U_c R_c^T, stored as k adjacent columns (vjp caches only).

    effect[n,K,c] = g_{n,K} . U_c R_c^T h_n = sum_j (g.u_cj)(r_cj.h) is the first-order output change of
    removing the whole piece along probe K; score = ||effect||_K / sqrt(K). With rank 1 this is exactly
    the rank-one score |r.h| ||g.u|| / sqrt(K).
    """
    if pieces.read.shape[1] % rank:
        raise ValueError('column count must be a multiple of the piece rank')
    auto = torch.is_autocast_enabled(pieces.read.device.type)
    if entry['kind'] == 'scalar_loss':
        # One cotangent per example (GIM / scalar CE): score = |g . U_c R_c^T h|, exactly |r.h||g.u| at rank 1.
        h, g = entry['h'], entry['events']['g']
        if not auto: h, g = h.to(pieces.read.dtype), g.to(pieces.write.dtype)
        read = norm_input(h @ pieces.read).unflatten(-1, (-1, rank))       # [n, C, k]
        sens = norm_input(g @ pieces.write).unflatten(-1, (-1, rank))      # [n, C, k]
        return (read * sens).sum(-1).abs()
    if entry['kind'] != 'vjp':
        raise ValueError('rank-k pieces need a scalar_loss (gim) or vjp (fisher) cache')
    h, cot = entry['h'], entry['events']['cot']
    if not auto: h, cot = h.to(pieces.read.dtype), cot.to(pieces.write.dtype)
    read = (h @ pieces.read).unflatten(-1, (-1, rank))                   # [n, C, k]
    sens = (cot @ pieces.write).unflatten(-1, (-1, rank))                # [n, K, C, k]
    effect = torch.einsum('nck,nqck->nqc', read, sens)
    return torch.linalg.vector_norm(norm_input(effect), dim=1) / math.sqrt(entry["output_count"])


def signed_piece_effects(entry, pieces):
    """Signed first-order output effect of each rank-one piece: [n, N] (scalar_loss: GIM) or [n, K, N] (vjp: Fisher)."""
    auto = torch.is_autocast_enabled(pieces.read.device.type)
    h = entry['h'] if auto else entry['h'].to(pieces.read.dtype)
    if entry['kind'] == 'scalar_loss':
        g = entry['events']['g'] if auto else entry['events']['g'].to(pieces.write.dtype)
        return (h @ pieces.read) * (g @ pieces.write)
    if entry['kind'] == 'vjp':
        cot = entry['events']['cot'] if auto else entry['events']['cot'].to(pieces.write.dtype)
        return (h @ pieces.read).unsqueeze(1) * (cot @ pieces.write)
    raise ValueError('owned components need a scalar_loss (gim) or vjp (fisher) cache')


def _component_reduce(entry, effect):
    effect = norm_input(effect)
    if entry['kind'] == 'scalar_loss':
        return effect.abs()
    return torch.linalg.vector_norm(effect, dim=1) / math.sqrt(entry["output_count"])


def soft_component_scores(entry, pieces, ownership):
    """Component scores with soft ownership [N, M] (rows sum to 1): effect_m = sum_c A_cm z_c."""
    return _component_reduce(entry, signed_piece_effects(entry, pieces) @ ownership.to(pieces.read.dtype))


def hard_component_scores(entry, pieces, owner, components):
    """Component scores with hard ownership: owner [N] in 0..components-1."""
    z = norm_input(signed_piece_effects(entry, pieces))
    effect = torch.zeros(*z.shape[:-1], components, dtype=z.dtype, device=z.device).index_add_(z.dim()-1, owner, z)
    return _component_reduce(entry, effect)


def scores(entry, pieces):
    if entry['kind'] == 'scalar_ig':
        return integrated_scalar_scores(entry, pieces.read, pieces.write)
    if entry['kind'] in ('pile_last_up', 'pile_last_down'):
        from parauto_pile_last import scores as last_scores
        return last_scores(entry, pieces)
    if entry['kind'] == 'scalar_loss':
        # One exact scalar-loss cotangent per example; no random probes or RMS.
        auto = torch.is_autocast_enabled(pieces.read.device.type)
        h = entry['h'] if auto else entry['h'].to(pieces.read.dtype)
        g = entry['events']['g'] if auto else entry['events']['g'].to(pieces.write.dtype)
        read = h @ pieces.read
        write = g @ pieces.write
        return norm_input(read).abs() * norm_input(write).abs()
    if entry['kind'] == 'finite_ablation':
        from finite_ablation import scores as finite_scores
        return finite_scores(entry, pieces)
    if entry['kind'] == 'write_norm':
        # ||u_c (r_c.h)||_2 = |r_c.h| ||u_c||: size of the piece's write at this
        # matrix output, in every output direction. Needs no loss cotangent.
        h = entry['h'] if torch.is_autocast_enabled(pieces.read.device.type) else entry['h'].to(pieces.read.dtype)
        read = h @ pieces.read
        return norm_input(read).abs() * norm_input(pieces.write).square().sum(0).add(1e-30).sqrt()
    if entry['kind'] == 'metric_interact':
        # Interaction-aware score in the fixed metric M = L L^T (static_fisher.py): with v_c = a_c L^T u_c and
        # y = sum_c v_c (the matrix's whole output, L^T W h, since the pieces sum to W), 2 v_c.y - ||v_c||^2 is how much
        # keeping c alone reduces ||y - kept||^2; it sees overlap and cancellation with all other pieces through y.
        # In KL units (squared), may be negative; computed in fp32 because of the cancellation.
        with torch.autocast(pieces.read.device.type, enabled=False):
            read, write = pieces.read.float(), pieces.write.float()
            a = entry['h'].float() @ read                                      # [n, C]
            basis = entry['shared']['metric'].float().T @ write                # L^T U [rank, C]
            w = (a @ write.T) @ entry['shared']['metric'].float()              # L^T (sum_c a_c u_c), as a row [n, rank]
            return 2 * a * (w @ basis) - a.square() * basis.square().sum(0)
    if entry['kind'] == 'metric_write':
        # |r_c.h| sqrt(u_c^T M u_c): the write's size in a fixed output metric M = L L^T (static_fisher.py: the
        # target's token-averaged output Fisher), shared['metric'] = L [d_out, rank]. Needs no loss cotangent.
        auto = torch.is_autocast_enabled(pieces.read.device.type)
        h = entry['h'] if auto else entry['h'].to(pieces.read.dtype)
        metric = entry['shared']['metric']
        read = h @ pieces.read
        write = metric.T @ pieces.write if auto else metric.T.to(pieces.write.dtype) @ pieces.write
        return norm_input(read).abs() * norm_input(write).square().sum(0).add(1e-30).sqrt()
    h = entry['h']
    if entry['kind'] == 'vjp' and not torch.is_autocast_enabled(pieces.read.device.type): h = h.to(pieces.read.dtype)
    read = h @ pieces.read
    norm = squared_sensitivity(entry, pieces.write).add(1e-30).sqrt()
    return norm_input(read).abs() * norm / math.sqrt(entry["output_count"])


def integrated_scalar_scores(entry, read, write):
    """Signed quadrature first, magnitude second; caches are fixed during fitting."""
    automatic = torch.is_autocast_enabled(read.device.type)
    total = None
    for point, weight in enumerate(entry['shared']['quadrature_weights']):
        h, g = entry['h'][:, point], entry['events']['g'][:, point]
        if not automatic: h, g = h.to(read.dtype), g.to(write.dtype)
        value = norm_input(h @ read) * norm_input(g @ write) * weight
        total = value if total is None else total + value
    return total.abs()


@torch.no_grad()
def cache_events(target, x, chunk=256):
    if len(target.layers) != 2:
        raise ValueError("Structured cache currently supports the two-layer residual toy")
    dtype = x.dtype
    a0, a1 = [layer.mlp_in.weight.detach() for layer in target.layers]
    b0, b1 = [layer.mlp_out.weight.detach() for layer in target.layers]
    f = target.W_U.detach().T
    h0 = x @ target.W_E.detach()
    z0 = h0 @ a0.T
    r0 = z0.relu()
    h1 = h0 + r0 @ b0.T
    z1 = h1 @ a1.T
    r1 = z1.relu()
    mask0, mask1 = z0 > 0, z1 > 0
    output = (h1 + r1 @ b1.T) @ f.T
    # QR is a frozen cache transformation, not part of the training graph.
    # Float64 setup keeps the orthogonal change of coordinates accurate.
    fd, b0d, b1d = f.double(), b0.double(), b1.double()
    p, q = fd @ b0d, fd @ b1d
    width = b0.shape[1]
    if b1.shape[1] != width:
        raise ValueError("The two toy MLPs must have equal hidden widths")
    _, packed = torch.linalg.qr(torch.cat((p, q), dim=1), mode="reduced")
    p_small, q_small = packed[:, :width], packed[:, width:]
    coupling = a1.double() @ b0d
    factors = []
    for start in range(0, len(x), chunk):
        d0, d1 = mask0[start:start+chunk], mask1[start:start+chunk]
        j = (p_small + (q_small.unsqueeze(0) * d1.unsqueeze(1)) @ coupling) * d0.unsqueeze(1)
        factors.append(torch.linalg.qr(j, mode="reduced").R.to(dtype))
    # A complete orthonormal output basis separates the downstream MLP's
    # range from its orthogonal complement. Squared norms then ADD; there is
    # no subtraction of large quadratic terms near a null write direction.
    output_basis, response = torch.linalg.qr(q, mode="complete")
    rank = min(q.shape)
    entries = [
        {"kind": "input_qr", "h": h0, "events": {"factor": torch.cat(factors)}, "shared": {}},
        {"kind": "output_residual", "h": r0, "events": {"mask": mask1},
         "shared": {"output": (output_basis.T @ fd).to(dtype),
                    "response": response[:rank].to(dtype), "read": a1}},
        {"kind": "input_mask", "h": h1, "events": {"mask": mask1},
         "shared": {"factor": response[:rank].to(dtype)}},
        {"kind": "output_shared", "h": r1, "events": {}, "shared": {"output": f}},
    ]
    # Fixed net effects support scoring validation and analysis. Compute
    # them analytically once, then discard all temporary output sensitivities.
    for i, (entry, weight) in enumerate(zip(entries, (a0, b0, a1, b1), strict=True)):
        v = entry["h"] @ weight.T
        if i == 0:
            v = v * mask0
            effect = v @ p.T.to(dtype) + ((v @ coupling.T.to(dtype)) * mask1) @ q.T.to(dtype)
        elif i == 1:
            effect = v @ f.T + ((v @ a1.T) * mask1) @ q.T.to(dtype)
        elif i == 2:
            effect = (v * mask1) @ q.T.to(dtype)
        else:
            effect = v @ f.T
        entry.update(structured=True, output_count=f.shape[0], net_rms=effect.square().mean(-1).sqrt())
    names = ("layer0.mlp_in", "layer0.mlp_out", "layer1.mlp_in", "layer1.mlp_out")
    return {"x": x, "output": output, "matrices": dict(zip(names, entries, strict=True))}


def cache_bytes(cache):
    """Count distinct tensor storages, including shared activation/mask data."""
    seen = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            seen[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
    visit(cache)
    return sum(seen.values())
