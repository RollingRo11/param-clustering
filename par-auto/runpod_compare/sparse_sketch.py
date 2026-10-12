"""Signed sketch backward through the SPARSE frozen-gate circuit, on cached activations only.

This is the consumer-gated idea with signs kept. Removing piece c changes the k sketch directions at
the final residual by e_c = rho_c * g_c, where g_c is found by propagating the sketch cotangents
backward through the circuit made of the pieces that are currently ON. Keeping signs makes every step
factorize through a shared k x d cotangent, so nothing has to be materialized per piece pair:

    cotangent at a site   Gamma [n, k, d]
    piece effect          g_c = Gamma u_c                       one matmul per site
    backward through ON   Gamma_next = sum_{c on} (Gamma u_c) r_c^T = (Gamma U_on) R_on^T

so the per-token GELU derivative, the per-head cross-position attention transport and the softmax
Jacobian for q/k pieces are all affordable and exact, which the |K| form could not do.

Two rule sets, because the backward pass is written by hand here:
  rules='exact'  true gradients: exact RMSNorm Jacobian, ordinary softmax Jacobian, no scaling, and the
                 probes are k labels sampled from the model. This is the mode the finite-difference
                 check verifies, because only here is the circuit's backward the model's own.
  rules='gim'    GIM's surrogate rules, applied at every step we compute ourselves, matching
                 gim_attribution.RULES: temperature-2 softmax in the Jacobian, q/k gradients x0.25,
                 v x0.5, and a frozen RMS scale (the sigma term dropped). One probe, the true next
                 token's cross entropy. A finite-difference check must FAIL in this mode by design:
                 GIM is deliberately not the gradient.

What is exact in both: GELU derivative at the cached pre-activation, attention probabilities, rotary
applied to the q/k piece writes, causal masking, and the restriction of every backward step to the ON
set. What is approximate: the attention pattern is held at its intact value, so pieces move it only
through the softmax Jacobian (first order).

Score convention matches the Fisher sensor: a_c = |rho_c| * ||g_c||_2 / sqrt(k).

Two points to evaluate at. By default the cache is the INTACT model: every activation, GELU slope,
RMSNorm scale and attention pattern is the dense model's, and the on-set only decides which pieces
carry cotangents. With `masks` the cache comes from the MASKED forward instead (each matrix replaced
by the sum of its on pieces, the same hook the KLKeep evaluation uses), so rho, slopes, patterns and
the probe cotangents are all the sparse model's own. `iterate_sparse` rebuilds that cache every time
the on-set moves.
"""
import contextlib
import math

import torch
import torch.nn.functional as F

from common import MODULES

KINDS = ('attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'mlp.c_fc', 'mlp.down_proj')


def gelu_derivative(z):
    """Derivative of the tanh-approximate GELU this target uses, in FP32; keeps the negative region."""
    inner = 0.7978845608028654 * (z + 0.044715 * z.pow(3))
    tanh = torch.tanh(inner)
    return .5*(1 + tanh) + .5*z*(1 - tanh.square())*0.7978845608028654*(1 + 3*0.044715*z.square())


def rms_vjp(cotangent, x, gamma, eps, frozen=False):
    """Backward of y = gamma * x / sqrt(mean(x^2) + eps) (RMSNorm: no mean subtraction).

    cotangent: [..., k, d]; x: [..., 1, d]. frozen=True is GIM's rule: the scale is treated as a
    constant, which drops the second term.
    """
    d = x.shape[-1]
    sigma = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)                 # [..., 1, 1]
    weighted = cotangent * gamma                                             # [..., k, d]
    if frozen:
        return weighted/sigma
    inner = (weighted * x).sum(-1, keepdim=True)                             # [..., k, 1]
    return weighted/sigma - x * inner/(d * sigma.pow(3))


def masked_hooks(model, pieces, masks):
    """Forward hooks replacing each matrix's output by its on pieces: ((h @ read) * mask) @ write^T.

    Byte-for-byte the expression consumer_probe's KLKeep evaluation uses, so a circuit built under these
    hooks is cached at exactly the sparse model that gets evaluated. Caller removes the handles.
    """
    handles = []
    for name, p in zip(MODULES, pieces):
        def replace(layer, inputs, output, p=p, mask=masks[name]):
            value = ((inputs[0] @ p.read)*mask) @ p.write.T
            return value if layer.bias is None else value+layer.bias
        handles.append(model.get_submodule(name).register_forward_hook(replace))
    return handles


class Circuit:
    """Cached state of one batch of sequences, everything the backward pass needs.

    masks=None caches the intact model; masks={site: [B,T,C] 0/1} caches the masked model.
    labels=[B*T, k] reuses given probe labels (exact rules) instead of sampling new ones.
    """

    def __init__(self, model, rows, pieces, probe_k=8, seed=0, rules='exact', labels=None, autocast=True, masks=None):
        from gim_attribution import RULES, gim_context
        assert rules in ('exact', 'gim')
        # FP32 forwards are needed when the cotangents are compared against finite differences.
        self.cast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if autocast else contextlib.nullcontext
        self.rules = rules
        self.scale = dict(q_proj=RULES['q_scale'], k_proj=RULES['k_scale'], v_proj=RULES['v_scale']) if rules == 'gim' else {}
        self.frozen_rms = rules == 'gim' and RULES['freeze_rms']
        self.temperature = RULES['softmax_temperature'] if rules == 'gim' else 1.
        self.model = model; self.pieces = pieces
        self.index = {n: i for i, n in enumerate(MODULES)}
        ids = rows[:, :-1]
        self.batch, self.length = ids.shape
        attn = model.get_submodule('h.0.attn')
        assert attn.n_head == attn.n_key_value_heads, 'grouped-query attention is not handled'
        self.heads, self.head_dim = attn.n_head, attn.head_dim
        self.masks = masks
        pre, out, hooks = {}, {}, []
        try:
            if masks is not None:
                # Registered first: a later hook receives the output an earlier hook returned.
                hooks.extend(masked_hooks(model, pieces, masks))
            def retain(name):
                # A forward hook that returns anything but None REPLACES the module output.
                def hook(_module, inputs, output):
                    pre[name] = inputs[0].detach()
                    out[name] = output.detach()
                    return None
                return hook
            for name in MODULES + tuple(f'h.{l}.rms_{j}' for l in range(4) for j in (1, 2)):
                hooks.append(model.get_submodule(name).register_forward_hook(retain(name)))
            held = {}
            def detach_input(_module, inputs):
                value = inputs[0].detach().requires_grad_()
                held['resid'] = value
                return (value,)
            hooks.append(model.ln_f.register_forward_pre_hook(detach_input))
            # GIM's rules also change the backward from the loss to this site (frozen RMS in ln_f).
            scope = gim_context(model, temperature=self.temperature, freeze_rms=self.frozen_rms) if rules == 'gim' \
                else contextlib.nullcontext()
            with scope:
                with torch.enable_grad(), self.cast():
                    logits = model(ids)[0].flatten(0, 1)
                if rules == 'gim':
                    # One probe: the true next token, as GIM uses.
                    probe_k = 1
                    labels = rows[:, 1:].flatten()[:, None]
                elif labels is not None:
                    labels = labels.to(logits.device)
                    probe_k = labels.shape[1]
                else:
                    generator = torch.Generator(device=logits.device).manual_seed(seed)
                    labels = torch.multinomial(logits.detach().float().softmax(-1), probe_k, replacement=True, generator=generator)
                probes = []
                for j in range(probe_k):
                    with self.cast():
                        loss = F.cross_entropy(logits, labels[:, j], reduction='sum')
                    probes.append(torch.autograd.grad(loss, held['resid'], retain_graph=j < probe_k-1)[0].detach().float())
            self.final = torch.stack(probes, -2)                              # [B, T, k, d_model]
            self.labels = labels                                              # probe labels (sampled, or the true next token)
            del logits, held, probes
        finally:
            for hook in hooks: hook.remove()
        self.probe_k = probe_k
        self.pre = {n: pre[n].float() for n in pre}
        self.out = {n: out[n].float() for n in out}
        self.activation = {}
        for name in MODULES:
            read = pieces[self.index[name]].read.float()
            self.activation[name] = self.pre[name].flatten(0, 1) @ read       # rho [n, C]
        self.pattern, self.backward_pattern, self.values, self.rotated = {}, {}, {}, {}
        for layer in range(4):
            module = model.get_submodule(f'h.{layer}.attn')
            q, k = self._rotate(module, self.out[f'h.{layer}.attn.q_proj'], self.out[f'h.{layer}.attn.k_proj'])
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(module.bias[:, :, :self.length, :self.length] == 0, float('-inf'))
            self.pattern[layer] = scores.softmax(-1)                          # [B, H, T, T], forward values
            # GIM replaces the softmax VJP's probabilities with those at temperature T.
            self.backward_pattern[layer] = (scores/self.temperature).softmax(-1) if self.temperature != 1. else self.pattern[layer]
            self.values[layer] = self._heads(self.out[f'h.{layer}.attn.v_proj'])
            self.rotated[layer] = (q, k)

    def _heads(self, value):
        return value.view(self.batch, self.length, self.heads, self.head_dim).transpose(1, 2)

    def _rotate(self, module, q_flat, k_flat):
        q, k = self._heads(q_flat), self._heads(k_flat)
        position = torch.arange(self.length, device=q.device).unsqueeze(0).clamp(max=module.n_ctx-1)
        cos, sin = module.rotary_cos[position].to(q.dtype), module.rotary_sin[position].to(q.dtype)
        return module.apply_rotary_pos_emb(q, k, cos, sin)

    @staticmethod
    def rotate_every_two(x):
        """The target's half-split rotation (rotary_adjacent_pairs is False)."""
        half = x.shape[-1]//2
        return torch.cat((-x[..., half:], x[..., :half]), -1)

    def rotary(self, layer):
        module = self.model.get_submodule(f'h.{layer}.attn')
        position = torch.arange(self.length, device=self.final.device).clamp(max=module.n_ctx-1)
        return module.rotary_cos[position].float(), module.rotary_sin[position].float()


def contract_rotated(cotangent, write, cos, sin):
    """sum_d cotangent[b,t,k,d] * rope(write, t)[d,c], with rope applied to the piece write itself."""
    turned = Circuit.rotate_every_two(write.T).T                                  # [dh, C]
    return (torch.einsum('btkd,dc->btkc', cotangent*cos[None, :, None, :], write)
            + torch.einsum('btkd,dc->btkc', cotangent*sin[None, :, None, :], turned))


def backward_effects(circuit, on):
    """Signed sketch effect per unit activation, g_c [B, T, k, C], for every site.

    `on` maps site -> [B, T, C] 0/1: only those pieces carry cotangents backward, which is what makes
    this a sensor at the sparse point. Every piece still gets its own effect read off the cotangent at
    its site, whether or not it is on.
    """
    model, pieces, index = circuit.model, circuit.pieces, circuit.index
    B, T, k = circuit.batch, circuit.length, circuit.probe_k
    heads, head_dim = circuit.heads, circuit.head_dim
    stream = circuit.final                                                        # [B,T,k,d_model]
    effects = {}
    for layer in reversed(range(4)):
        down, cfc = f'h.{layer}.mlp.down_proj', f'h.{layer}.mlp.c_fc'
        write_down, read_down = pieces[index[down]].write.float(), pieces[index[down]].read.float()
        effects[down] = stream @ write_down
        gelu_side = (effects[down] * on[down].unsqueeze(2)) @ read_down.T          # [B,T,k,d_hidden]
        pre_activation = circuit.out[cfc].view(B, T, 1, -1)
        hidden = gelu_side * gelu_derivative(pre_activation)
        write_cfc, read_cfc = pieces[index[cfc]].write.float(), pieces[index[cfc]].read.float()
        effects[cfc] = hidden @ write_cfc
        norm = model.get_submodule(f'h.{layer}.rms_2')
        stream = stream + rms_vjp((effects[cfc] * on[cfc].unsqueeze(2)) @ read_cfc.T,
                                  circuit.pre[f'h.{layer}.rms_2'].view(B, T, 1, -1),
                                  norm.weight.float(), norm.variance_epsilon, circuit.frozen_rms)

        out_site = f'h.{layer}.attn.o_proj'
        write_o, read_o = pieces[index[out_site]].write.float(), pieces[index[out_site]].read.float()
        effects[out_site] = stream @ write_o
        attended = ((effects[out_site] * on[out_site].unsqueeze(2)) @ read_o.T
                    ).view(B, T, k, heads, head_dim).permute(0, 3, 1, 2, 4)        # [B,H,T,k,dh]
        pattern, values = circuit.pattern[layer], circuit.values[layer]
        query_rot, key_rot = circuit.rotated[layer]
        cos, sin = circuit.rotary(layer)
        v_site, q_site, k_site = (f'h.{layer}.attn.{n}_proj' for n in ('v', 'q', 'k'))
        # A v piece at key position s is consumed by every query position that attends to it.
        carried = torch.einsum('bhts,bhtkd->bhskd', pattern, attended)
        write_v = pieces[index[v_site]].write.float().view(heads, head_dim, -1)
        effects[v_site] = torch.einsum('bhskd,hdc->bskc', carried, write_v)
        del carried
        write_q = pieces[index[q_site]].write.float().view(heads, head_dim, -1)
        write_k = pieces[index[k_site]].write.float().view(heads, head_dim, -1)
        effects[q_site] = torch.zeros(B, T, k, write_q.shape[-1], device=stream.device)
        effects[k_site] = torch.zeros(B, T, k, write_k.shape[-1], device=stream.device)
        scale = 1./math.sqrt(head_dim)
        for head in range(heads):                                                 # per head: bounds memory
            # Cotangent of the attention scores, then through the softmax.
            score_side = torch.einsum('btkd,bsd->btsk', attended[:, head], values[:, head])
            # The softmax VJP uses GIM's temperature-T probabilities; the forward pattern is unchanged.
            weight = circuit.backward_pattern[layer][:, head].unsqueeze(-1)        # [B,T,T,1]
            logit_side = weight * (score_side - (weight*score_side).sum(2, keepdim=True))
            del score_side
            for_query = torch.einsum('btsk,bsd->btkd', logit_side, key_rot[:, head].float())*scale
            for_key = torch.einsum('btsk,btd->bskd', logit_side, query_rot[:, head].float())*scale
            del logit_side
            effects[q_site] += contract_rotated(for_query, write_q[head], cos, sin)
            effects[k_site] += contract_rotated(for_key, write_k[head], cos, sin)
            del for_query, for_key
        del attended
        # GIM scales the gradient at the q/k/v projection outputs, which scales both the piece effect
        # and everything that continues backward through those pieces.
        for site, key in ((q_site, 'q_proj'), (k_site, 'k_proj'), (v_site, 'v_proj')):
            if key in circuit.scale:
                effects[site] = effects[site] * circuit.scale[key]
        norm = model.get_submodule(f'h.{layer}.rms_1')
        side = None
        for site in (q_site, k_site, v_site):
            term = (effects[site] * on[site].unsqueeze(2)) @ pieces[index[site]].read.float().T
            side = term if side is None else side + term
        stream = stream + rms_vjp(side, circuit.pre[f'h.{layer}.rms_1'].view(B, T, 1, -1),
                                  norm.weight.float(), norm.variance_epsilon, circuit.frozen_rms)
    return effects


def sketch_scores(circuit, on):
    """a_c = |rho_c| * ||g_c||_2 / sqrt(k), the same units as the Fisher sensor."""
    effects = backward_effects(circuit, on)
    scores = []
    for name in MODULES:
        value = torch.linalg.vector_norm(effects[name], dim=2).flatten(0, 1)/math.sqrt(circuit.probe_k)
        scores.append(circuit.activation[name].abs() * value)
    return scores, effects


def iterate(circuit, keep, fallback, iterations=3):
    """Fixed point: rank, take the top-`keep` per token as the on-set, rescore through that circuit."""
    counts = [p.count for p in circuit.pieces]
    scores, churn = [f.clone() for f in fallback], []
    previous = None
    for _ in range(iterations):
        combined = torch.cat(scores, -1)
        chosen = combined.argsort(dim=-1, descending=True, stable=True)[..., :max(1, int(keep))]
        mask = torch.zeros_like(combined).scatter_(-1, chosen, 1.)
        if previous is not None:
            churn.append(float((mask != previous).float().sum(-1).mean()/2))
        previous = mask
        on = {n: m.view(circuit.batch, circuit.length, -1) for n, m in zip(MODULES, mask.split(counts, -1))}
        scores, _ = sketch_scores(circuit, on)
    return scores, dict(on_set_churn=churn, keep=int(keep), iterations=iterations)


def on_set(scores, keep, batch, length, counts):
    """Top-`keep` pieces per token across all sites -> ({site: [B,T,C] 0/1}, flat [n, sum C] mask)."""
    combined = torch.cat(scores, -1)
    chosen = combined.argsort(dim=-1, descending=True, stable=True)[..., :max(1, int(keep))]
    mask = torch.zeros_like(combined).scatter_(-1, chosen, 1.)
    return {n: m.view(batch, length, -1) for n, m in zip(MODULES, mask.split(counts, -1))}, mask


def iterate_sparse(model, rows, pieces, keep, fallback, iterations=3, probe_k=8, seed=0, rules='exact',
                   labels=None, autocast=True):
    """Same fixed point as `iterate`, but every rescoring rebuilds the cache from the masked forward.

    Round i: take the top-`keep` per token of the current scores as the on-set, run the masked model
    (activations, GELU slopes, RMS scales, attention patterns and probe cotangents all at the sparse
    point), push the sketch backward through the on pieces, rescore every piece by |rho| * ||g|| at
    that point. `labels` pins the probes to the ones the intact circuit sampled, so the only thing
    that differs from `iterate` is where the circuit is evaluated.
    """
    counts = [p.count for p in pieces]
    batch, length = rows.shape[0], rows.shape[1]-1
    scores, churn, previous = [f.clone() for f in fallback], [], None
    for _ in range(iterations):
        on, mask = on_set(scores, keep, batch, length, counts)
        if previous is not None:
            churn.append(float((mask != previous).float().sum(-1).mean()/2))
        previous = mask
        circuit = Circuit(model, rows, pieces, probe_k=probe_k, seed=seed, rules=rules, labels=labels,
                          autocast=autocast, masks=on)
        labels = circuit.labels
        scores, _ = sketch_scores(circuit, on)
        del circuit
    return scores, dict(on_set_churn=churn, keep=int(keep), iterations=iterations)
