"""Importance by who reads you, not by the slope of the loss.

A gradient sensor asks "if I nudge this piece, how much does the loss move" — a question about the
dense model at the intact point. This asks a circuit question instead: does anything that is itself
ON downstream actually consume this piece's write? In a residual network the "who reads whom"
structure is static and precomputable, one small matmul per pair of matrices:

    K[c', c] = r_c' . (what piece c writes, expressed where c' reads)

Per token, importance runs backward through that graph, through ON pieces only:

    G_c = direct_c + sum_{c' on, downstream} |K[c', c]| G_c',     a_c(x) = |rho_c(x)| G_c(x)

G_c is importance per unit of activation. Two properties follow. It is a sparse-point sensor with no
sparse forward pass: a write consumed only by pieces you already removed contributes nothing. And the
gate is hard, not differential: a write into a direction no active reader picks up scores zero however
steep the loss is along it at the intact point.

The graph for this target (per layer: q,k,v read rms_1; o writes the stream; c_fc reads rms_2; down
writes the stream):
  - o_proj, down_proj   write the residual stream. Direct term: the size of that write in the
                        averaged output metric M = E[g g^T] at the final pre-norm residual.
  - v_proj              is read by o_proj of its own layer:      K = R_o^T U_v
  - c_fc                is read by down_proj of its own layer:   K = R_down^T (gate * U_cfc), gate the
                        mean GELU derivative from the cache
  - residual writers    are read by q/k/v/c_fc of later blocks:  K = (gamma * R_reader)^T U_producer,
                        times that token's RMSNorm scale
  - q_proj, k_proj      have no linear consumer: they modulate the attention pattern. v1 does NOT
                        recompute patterns, so these pieces fall back to the supplied sensor scores.

Averaged, not per token: the Fisher metric M, the GELU gate, and the attention mixing weight between
v_proj and o_proj (taken as 1). Per token: rho, the on-sets, and the RMSNorm scales.
"""
import torch
import torch.nn.functional as F

from common import MODULES

RESIDUAL_WRITERS = ('attn.o_proj', 'mlp.down_proj')
PATTERN_ONLY = ('attn.q_proj', 'attn.k_proj')


def site_kind(name):
    return name.split('.', 2)[2]


def site_layer(name):
    return int(name.split('.')[1])


STREAM_READERS = ('attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'mlp.c_fc')


def reads_after(producer, reader):
    """True when `reader` reads a residual stream that already contains `producer`'s write."""
    pk, rk = site_kind(producer), site_kind(reader)
    if pk not in RESIDUAL_WRITERS or rk not in STREAM_READERS:
        return False
    pl, rl = site_layer(producer), site_layer(reader)
    if rl > pl:
        return True
    # Same block: the MLP reads the post-attention stream; nothing reads its own block's MLP write.
    return rl == pl and pk == 'attn.o_proj' and rk == 'mlp.c_fc'


def norm_of(name):
    """The norm whose output this matrix reads."""
    return f"h.{site_layer(name)}." + ('rms_2' if site_kind(name) == 'mlp.c_fc' else 'rms_1')


@torch.no_grad()
def mean_gelu_gate(model, rows_iter):
    """Mean GELU derivative per hidden unit, from the cache: the gate through which c_fc is read."""
    totals, count = {}, 0
    for rows in rows_iter:
        pre, hooks = {}, []
        try:
            for layer in range(4):
                name = f'h.{layer}.mlp.c_fc'
                hooks.append(model.get_submodule(name).register_forward_hook(
                    lambda _m, _i, out, n=name: pre.__setitem__(n, out.detach())))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(rows[:, :-1])
        finally:
            for hook in hooks: hook.remove()
        tokens = 0
        for name, value in pre.items():
            z = value.float().flatten(0, 1)
            # Derivative of the tanh-approximate GELU the target uses, in FP32.
            inner = 0.7978845608028654 * (z + 0.044715 * z.pow(3))
            tanh = torch.tanh(inner)
            gate = .5*(1 + tanh) + .5*z*(1 - tanh.square())*0.7978845608028654*(1 + 3*0.044715*z.square())
            totals[name] = gate.sum(0) if name not in totals else totals[name] + gate.sum(0)
            tokens = z.shape[0]
        count += tokens
    return {n: v/count for n, v in totals.items()}, count


def final_metric(model, rows_iter, k=8, seed=7_000_001):
    """M = E[g g^T] at the residual just before the final norm, g from labels sampled from the model."""
    total, samples = None, 0
    generator = torch.Generator(device='cuda').manual_seed(seed)
    for rows in rows_iter:
        held, hooks = {}, []
        try:
            def detach_input(_module, inputs):
                value = inputs[0].detach().requires_grad_()
                held['resid'] = value
                return (value,)
            hooks.append(model.ln_f.register_forward_pre_hook(detach_input))
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(rows[:, :-1])[0].flatten(0, 1)
            labels = torch.multinomial(logits.detach().float().softmax(-1), k, replacement=True, generator=generator)
            for j in range(k):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(logits, labels[:, j], reduction='sum')
                g = torch.autograd.grad(loss, held['resid'], retain_graph=j < k-1)[0].detach().float().flatten(0, 1)
                total = g.T @ g if total is None else total + g.T @ g
            samples += logits.shape[0]*k
            del logits, held
        finally:
            for hook in hooks: hook.remove()
    return total/samples, samples


def build_graph(model, pieces, rows_iter, fisher_k=8, dtype=torch.bfloat16):
    """Static couplings |K| and direct terms. rows_iter is consumed twice, so pass a list.

    Not decorated with no_grad: final_metric runs its own backward passes.
    """
    rows = list(rows_iter)
    gates, gate_tokens = mean_gelu_gate(model, rows)
    metric, metric_samples = final_metric(model, rows, k=fisher_k)
    return _couplings(model, pieces, gates, metric, dtype, metric_samples, gate_tokens, fisher_k)


@torch.no_grad()
def _couplings(model, pieces, gates, metric, dtype, metric_samples, gate_tokens, fisher_k):
    index = {n: i for i, n in enumerate(MODULES)}
    weights = {n: model.get_submodule(n).weight.detach().float() for n in MODULES}
    gamma = {f'h.{l}.rms_{j}': model.get_submodule(f'h.{l}.rms_{j}').weight.detach().float()
             for l in range(4) for j in (1, 2)}
    direct, couplings = {}, {}
    for name in MODULES:
        kind = site_kind(name)
        if kind in RESIDUAL_WRITERS:
            write = pieces[index[name]].write.float()                       # [d_model, C]
            direct[name] = ((metric @ write) * write).sum(0).clamp_min(0).sqrt()
        else:
            direct[name] = torch.zeros(pieces[index[name]].count, device=metric.device)
    for producer in MODULES:
        pk, pl = site_kind(producer), site_layer(producer)
        write = pieces[index[producer]].write.float()
        if pk == 'attn.v_proj':                                             # read by its own o_proj
            reader = f'h.{pl}.attn.o_proj'
            couplings[(reader, producer)] = (pieces[index[reader]].read.float().T @ write).abs().to(dtype)
        elif pk == 'mlp.c_fc':                                              # read by its own down_proj
            reader = f'h.{pl}.mlp.down_proj'
            gated = gates[producer][:, None] * write
            couplings[(reader, producer)] = (pieces[index[reader]].read.float().T @ gated).abs().to(dtype)
        elif pk in RESIDUAL_WRITERS:
            for reader in MODULES:
                if not reads_after(producer, reader): continue
                read = gamma[norm_of(reader)][:, None] * pieces[index[reader]].read.float()
                couplings[(reader, producer)] = (read.T @ write).abs().to(dtype)
    return dict(direct=direct, couplings=couplings, metric=metric, gates=gates,
                provenance=dict(metric_samples=metric_samples, gate_tokens=int(gate_tokens), fisher_k=fisher_k,
                                coupling_pairs=len(couplings), pattern_only=list(PATTERN_ONLY)))


def propagate(direct, couplings, on, scales, counts, order, injected=None):
    """Backward pass over the graph: G[site] = direct + sum over ON readers of |K| G_reader.

    direct: {site: [C]}; couplings: {(reader, producer): |K| [C_reader, C_producer]};
    on: {site: [n, C] 0/1}; scales: {site: [n, 1] RMSNorm scale of the stream that site reads};
    order: sites in reverse execution order; injected: {site: [n, C]} importance supplied from outside
    for sites whose consumers this graph does not model (q_proj/k_proj), so that a residual write read
    only by those pieces still carries their importance rather than zero.
    Pure tensor algebra, so it is unit-testable.
    """
    importance = {}
    for site in order:
        if injected is not None and site in injected:
            importance[site] = injected[site]
            continue
        value = direct[site][None, :].expand(next(iter(on.values())).shape[0], counts[site]).clone().float()
        for (reader, producer), coupling in couplings.items():
            if producer != site or reader not in importance: continue
            carried = importance[reader] * on[reader]
            if site_kind(producer) in RESIDUAL_WRITERS:
                carried = carried * scales[reader]
            value = value + (carried.to(coupling.dtype) @ coupling).float()
        importance[site] = value
    return importance


@torch.no_grad()
def consumer_scores(model, rows, pieces, graph, keep, fallback, iterations=3):
    """Per-token consumer-gated scores, list over MODULES of [n, C].

    `fallback` is a list of [n, C] scores from another sensor: it seeds the first on-set and supplies
    q_proj/k_proj, whose writes have no linear consumer in v1. `keep` is the number of pieces kept per
    token when deciding which pieces count as ON.
    """
    index = {n: i for i, n in enumerate(MODULES)}
    counts = {n: pieces[index[n]].count for n in MODULES}
    activations, scales = {}, {}
    pre, norms, hooks = {}, {}, []
    try:
        for name in MODULES:
            hooks.append(model.get_submodule(name).register_forward_hook(
                lambda _m, inputs, _out, n=name: pre.__setitem__(n, inputs[0].detach())))
        for layer in range(4):
            for j in (1, 2):
                name = f'h.{layer}.rms_{j}'
                hooks.append(model.get_submodule(name).register_forward_hook(
                    lambda _m, inputs, _out, n=name: norms.__setitem__(n, inputs[0].detach())))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(rows[:, :-1])
    finally:
        for hook in hooks: hook.remove()
    for name in MODULES:
        h = pre[name].float().flatten(0, 1)
        activations[name] = (h @ pieces[index[name]].read.float()).abs()          # |rho| [n, C]
        value = norms[norm_of(name)].float().flatten(0, 1)
        eps = model.get_submodule(norm_of(name)).variance_epsilon
        scales[name] = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)     # [n, 1]
    del pre, norms
    order = [f'h.{l}.{k}' for l in reversed(range(4))
             for k in ('mlp.down_proj', 'mlp.c_fc', 'attn.o_proj', 'attn.v_proj', 'attn.q_proj', 'attn.k_proj')]
    scores = [f.clone() for f in fallback]
    history = []
    for _ in range(iterations):
        combined = torch.cat(scores, -1)
        k = max(1, min(combined.shape[-1], int(keep)))
        chosen = torch.zeros_like(combined).scatter_(-1, combined.argsort(dim=-1, descending=True, stable=True)[..., :k], 1.)
        on = dict(zip(MODULES, chosen.split([counts[n] for n in MODULES], -1)))
        injected = {n: (fallback[i]/activations[n].clamp_min(1e-12)).clamp_min(0)
                    for i, n in enumerate(MODULES) if site_kind(n) in PATTERN_ONLY}
        importance = propagate(graph['direct'], graph['couplings'], on, scales, counts, order, injected)
        new = []
        for i, name in enumerate(MODULES):
            new.append(fallback[i] if site_kind(name) in PATTERN_ONLY else activations[name] * importance[name])
        moved = float((torch.cat(new, -1).argsort(-1, descending=True)[..., :k]
                       != combined.argsort(-1, descending=True)[..., :k]).float().mean())
        history.append(moved)
        scores = new
    return scores, dict(on_set_churn=history, keep=int(keep), iterations=iterations)
