"""Per-piece finite local ablation on the upstream 67M target.

Remove one piece c at one token t (subtract u_c a_c, a_c = r_c.h_t, from the matrix
output), recompute the nearest nonlinearity exactly, and score the norm of the change:

  c_fc               ||gelu(z_t) - gelu(z_t - u_c a_c)||            3072 hidden units at t
  q_proj             ||y_t(q_t) - y_t(q_t - u_c a_c)||             softmax row of t recomputed
  k_proj, v_proj     sqrt(sum_{i>=t} ||y_i - y_i'||^2)             key/value t changes every later query
  o_proj, down_proj  |a_c| ||u_c||                                 no nonlinearity before the residual

y is the attention output before o_proj (all heads). This matches KLKeep masking: a mask
removes a piece at one position of one matrix. Attention internals are recomputed in FP32
from the cached BF16 q, k, v, so intact and ablated values come from the same arithmetic.

k_proj uses the exact single-logit identity: if only logit j of query i moves by d,
y_i' - y_i = P_ij expm1(d) (v_j - y_i) / (1 - P_ij + P_ij e^d).
v_proj is linear: sum_i ||P_ij dv_j||^2 = a^2 sum_h s_hj ||u_ch||^2, s_hj = sum_i P_hij^2.
"""
import math
import torch
from torch.utils.checkpoint import checkpoint
from common import MODULES

HEADS, HEAD_DIM = 6, 128
_K = math.sqrt(2/math.pi)
TOKEN_CHUNK = 64      # q/k tokens per recomputed chunk: [64, 6, C, 512] FP32 is 0.8 GB
GELU_CHUNK = 256      # GELU kernels stay fused; the chunk only bounds a fallback


def _rotate_half(x):
    n = x.shape[-1]//2
    return torch.cat((-x[..., n:], x[..., :n]), -1)


def _rope(x, cos, sin):
    return x*cos + _rotate_half(x)*sin


@torch.no_grad()
def capture_finite(model, rows, tokens_per_sequence=None, generator=None, microbatch=4, autocast=True):
    """Forward only. tokens_per_sequence=None scores every token (evaluation)."""
    B, T = len(rows), rows.shape[1]-1
    attentions = [block.attn for block in model.h]
    assert all(not a.rotary_adjacent_pairs and a.rotary_dim == a.head_dim == HEAD_DIM and a.n_head == HEADS
               and a.n_key_value_heads == HEADS for a in attentions)
    wanted = {n for n in MODULES if n.endswith(('q_proj', 'k_proj', 'v_proj', 'c_fc'))}
    pre, post = {n: [] for n in MODULES}, {n: [] for n in wanted}
    hooks = []
    def retain(name):
        def hook(_m, inputs, out):
            pre[name].append(inputs[0])
            if name in wanted: post[name].append(out)
        return hook
    for name in MODULES: hooks.append(model.get_submodule(name).register_forward_hook(retain(name)))
    try:
        for first in range(0, B, microbatch):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                model(rows[first:first+microbatch, :-1])
    finally:
        for hook in hooks: hook.remove()
    store = torch.bfloat16 if autocast else torch.float32
    h = {n: torch.cat(v).flatten(0, 1).to(store) for n, v in pre.items()}
    out = {n: torch.cat(v) for n, v in post.items()}                          # [B, T, d]
    if tokens_per_sequence is None:
        seq = torch.arange(B, device='cuda').repeat_interleave(T)
        pos = torch.arange(T, device='cuda').repeat(B)
    else:
        pos = torch.stack([torch.randperm(T, device='cuda', generator=generator)[:tokens_per_sequence].sort().values
                           for _ in range(B)]).flatten()
        seq = torch.arange(B, device='cuda').repeat_interleave(tokens_per_sequence)
    flat = seq*T + pos
    layers = []
    causal = torch.ones(T, T, dtype=torch.bool, device='cuda').tril()
    for i, attention in enumerate(attentions):
        heads = lambda x: x.float().view(B, T, HEADS, HEAD_DIM).transpose(1, 2)     # [B, 6, T, 128]
        cos, sin = attention.rotary_cos[:T].float(), attention.rotary_sin[:T].float()
        q = _rope(heads(out[f'h.{i}.attn.q_proj']), cos, sin)
        k = _rope(heads(out[f'h.{i}.attn.k_proj']), cos, sin)
        v = heads(out[f'h.{i}.attn.v_proj'])
        with torch.autocast('cuda', enabled=False):
            P = ((q @ k.transpose(-1, -2))/math.sqrt(HEAD_DIM)).masked_fill(~causal, -math.inf).softmax(-1)
            y = P @ v
        layers.append(dict(q=q, k=k, v=v, P=P, y=y, s=P.square().sum(-2), cos=cos, sin=sin,
                           q16=q.bfloat16(), k16=k.bfloat16(), v16=v.bfloat16()))
    effects = dict(q_proj='q', k_proj='k', v_proj='v', c_fc='gelu', o_proj='write', down_proj='write')
    entries = []
    for name in MODULES:
        effect = effects[name.split('.')[-1]]
        events = dict(seq=seq, pos=pos)
        if effect == 'gelu': events['z'] = out[name].flatten(0, 1)[flat].float()
        entries.append(dict(structured=True, kind='finite_ablation', effect=effect, h=h[name][flat],
                            net_rms=torch.zeros(len(flat), device='cuda'), events=events,
                            shared=dict(sensor='finite_ablation', attn=layers[int(name.split('.')[1])])))
    return entries


# ---- c_fc: fused GELU difference, no [tokens, C, 3072] tensor in memory ----

def _gelu_pair(x):                           # gelu and its derivative from one tanh
    t = torch.tanh(_K*(x+0.044715*x*x*x))
    return 0.5*x*(1+t), 0.5*(1+t) + 0.5*x*(1-t*t)*_K*(1+3*0.044715*x*x)


def _gelu(x):
    return 0.5*x*(1+torch.tanh(_K*(x+0.044715*x*x*x)))


def _gelu_fwd(z, gz, a, u):                  # z, gz=gelu(z) [n,d]; a [n,C]; u [d,C] -> [n,C]
    return (gz[:, None, :] - _gelu(z[:, None, :] - a[:, :, None]*u.T[None])).square().sum(-1)


def _gelu_grad_a(z, gz, a, u, g):            # d(diff^2)/dw = 2 (gelu(z) - gelu(z-w)) gelu'(z-w), w = a u
    value, slope = _gelu_pair(z[:, None, :] - a[:, :, None]*u.T[None])
    return g*(2*(gz[:, None, :] - value)*slope*u.T[None]).sum(-1)


def _gelu_grad_u(zt, gzt, at, u, gt):        # zt, gzt [d,n], at/gt [C,n]: tokens innermost, so the sum fuses
    value, slope = _gelu_pair(zt[None] - at[:, None, :]*u.T[:, :, None])                # [C,d,n]
    return ((gt*at)[:, None, :]*2*(gzt[None] - value)*slope).sum(-1).T


_compiled = {}
def _kernel(fn):
    if fn not in _compiled: _compiled[fn] = torch.compile(fn, fullgraph=True)
    return _compiled[fn]


class GeluEffect(torch.autograd.Function):
    """Squared GELU change per (token, piece); FP32, chunked over tokens."""
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, z, a, u):
        gz = _gelu(z)
        ctx.save_for_backward(z, gz, a, u)
        return torch.cat([_kernel(_gelu_fwd)(z[i:i+GELU_CHUNK], gz[i:i+GELU_CHUNK], a[i:i+GELU_CHUNK], u)
                          for i in range(0, len(z), GELU_CHUNK)])

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, g):
        z, gz, a, u = ctx.saved_tensors
        g = g.float(); grad_a = grad_u = None
        chunks = [slice(i, i+GELU_CHUNK) for i in range(0, len(z), GELU_CHUNK)]
        if ctx.needs_input_grad[1]:
            grad_a = torch.cat([_kernel(_gelu_grad_a)(z[c], gz[c], a[c], u, g[c]) for c in chunks])
        if ctx.needs_input_grad[2]:
            t = lambda x: x.T.contiguous()
            grad_u = sum(_kernel(_gelu_grad_u)(t(z[c]), t(gz[c]), t(a[c]), u, t(g[c])) for c in chunks)
        return None, grad_a, grad_u


# ---- q and k: recompute the softmax, one token chunk at a time ----

def _q_change(M, a, P, valid, lowp):         # logit change -a M; M [m,6,C,T], P [m,6,T] -> dp [m,6,C,T]
    delta = (-(a/math.sqrt(HEAD_DIM))[:, None, :, None]*M.float()).masked_fill(~valid[:, None, None, :], -math.inf)
    weight = P[:, :, None]*torch.exp(delta - delta.amax(-1, keepdim=True))
    dp = weight/weight.sum(-1, keepdim=True).clamp_min(1e-30) - P[:, :, None]
    return dp.to(torch.bfloat16) if lowp else dp        # difference taken in FP32, stored for the matmul


def _k_change(M, a, P, D):                   # sum_{h,i} c^2 ||v_j - y_i||^2 -> [m,C]
    delta = (-(a/math.sqrt(HEAD_DIM))[:, None, :, None]*M.float()).clamp(-80, 80)
    P = P[:, :, None]
    c = P*torch.expm1(delta)/((1 - P) + P*torch.exp(delta))
    return (c.square()*D[:, :, None]).sum((1, 3))


def _attention_chunk(effect, attn, seq, pos, a, write, lowp, window):
    """a [m,C] FP32 codes at tokens (seq, pos); write [768,C]. Returns squared change [m,C].
    window: key range [0, w) for q (all keys a query can see); query range [w, T) for k.
    Heads-first [m,6,C,T] layout keeps every matmul a plain batched product (no transposed copies).
    lowp: BF16 matmuls (training/eval); the logit change and softmax difference stay FP32."""
    C = write.shape[1]
    u = write.reshape(HEADS, HEAD_DIM, C).transpose(1, 2)                                 # [6,C,128]
    ru = _rope(u[None], attn['cos'][pos][:, None, None], attn['sin'][pos][:, None, None])   # [m,6,C,128]
    x = (lambda name: attn[name+'16']) if lowp else (lambda name: attn[name])              # BF16 copies for matmuls
    if lowp: ru = ru.bfloat16()
    # M = rope(u).k (or .q) without the 1/sqrt(d) scale; the kernels apply it to a.
    if effect == 'q':
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=lowp):
            M = ru @ x('k')[seq, :, :window].transpose(-1, -2)                               # [m,6,C,w]
        valid = torch.arange(window, device='cuda')[None] <= pos[:, None]
        dp = _kernel(_q_change)(M, a, attn['P'][seq, :, pos, :window], valid, lowp)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=lowp):
            dy = dp @ x('v')[seq, :, :window]                                               # [m,6,C,128]
        return dy.float().square().sum((1, 3))
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=lowp):
        M = ru @ x('q')[seq, :, window:].transpose(-1, -2)                                  # [m,6,C,T-w]
    D = (attn['v'][seq, :, pos][:, :, None, :] - attn['y'][seq, :, window:]).square().sum(-1)  # [m,6,T-w]
    return _kernel(_k_change)(M, a, attn['P'][seq, :, window:, pos], D)


def scores(entry, pieces):
    auto = torch.is_autocast_enabled('cuda')
    h = entry['h'] if auto else entry['h'].to(pieces.read.dtype)
    a = (h @ pieces.read).float()                                                           # [n,C]
    write = pieces.write.float()
    effect, events = entry['effect'], entry['events']
    if effect == 'write':
        return a.abs()*write.square().sum(0).add(1e-30).sqrt()
    if effect == 'gelu':
        return GeluEffect.apply(events['z'], a, write).add(1e-30).sqrt()
    attn = entry['shared']['attn']
    if effect == 'v':
        s = attn['s'][events['seq'], :, events['pos']]                                     # [n,6]
        return a.abs()*(s @ write.reshape(HEADS, HEAD_DIM, -1).square().sum(1)).add(1e-30).sqrt()
    # Chunks of position-sorted tokens only touch the part of the causal window they need:
    # keys up to their last position (q), queries from their first position (k), in 64s.
    order = events['pos'].argsort(stable=True)
    T, sorted_pos = attn['P'].shape[-1], events['pos'][order].tolist()
    parts = []
    for i in range(0, len(a), TOKEN_CHUNK):
        idx = order[i:i+TOKEN_CHUNK]
        span = sorted_pos[i:i+TOKEN_CHUNK]
        window = min(T, -(-(span[-1]+1)//64)*64) if effect == 'q' else span[0]//64*64
        run = lambda a_, w_, seq=events['seq'][idx], pos=events['pos'][idx], window=window: \
            _attention_chunk(effect, attn, seq, pos, a_, w_, auto, window)
        # Recompute each chunk in backward instead of keeping [m,6,C,T] tensors for all chunks.
        parts.append(checkpoint(run, a[idx], write, use_reentrant=False) if a.requires_grad else run(a[idx], write))
    return torch.cat(parts)[order.argsort()].add(1e-30).sqrt()
