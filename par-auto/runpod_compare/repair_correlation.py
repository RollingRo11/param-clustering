"""Downstream self-repair, read off the residual stream's own bookkeeping. No ablation is run.

For a rank-one piece c that lives in a matrix whose output is added straight into the residual
stream, with read vector r_c, write vector u_c, and per-token activation

    rho_c(x) = r_c . h(x)                      h(x) = that matrix's input at token x

the piece writes rho_c(x) u_c into the stream. Everything the model adds after that site is

    later(x) = (residual just before ln_f) - (residual immediately after this site's output was added)

and the part of it that lies along this piece's own write direction is u_c . later(x). Then

    kappa_c = corr_x( rho_c(x),  u_c . later(x) )

Strongly negative kappa_c means the later layers write more against this piece exactly on the tokens
where it fires harder: they are reacting to it, so deleting the piece would be repaired too. A kappa
near zero together with a large |mean projection| is a constant cancellation, not a reactive one --
the later layers push back the same amount whatever this piece does. Both are reported, so the two
cases are never confused.

SCOPE. Only `h.{i}.attn.o_proj` and `h.{i}.mlp.down_proj` are measured. Those are the two matrices
per layer whose output vector IS a residual-stream vector, so u_c is a direction in the stream and
u_c . later is meaningful. For q_proj / k_proj / v_proj / c_fc the write lands in head space or in
the MLP hidden space, is passed through attention or GELU, and never enters the stream as itself;
there is no residual direction to project onto, so they are skipped rather than measured wrongly.

DEGENERATE SITE. For the last block's `mlp.down_proj` nothing is written afterwards, so later(x) is
identically zero and kappa is undefined. It is reported as NaN with zero projection variance, not
as a number.

NUMERICS. Streaming sums only: sum x, sum y, sum x^2, sum y^2, sum xy, count -- O(C) memory, no
token is kept. Per-batch reductions run in FP32; the running totals are FP64, and both series are
shifted by the first batch's mean before accumulating. Correlation is shift-invariant, so this
changes nothing mathematically but removes the catastrophic cancellation that a large mean
projection (the constant-cancellation case, which is exactly what we expect) would otherwise cause
in `sum y^2 - n * mean_y^2`. The model runs under bf16 autocast with no gradients and is never
modified: every quantity comes from forward hooks that only observe.
"""
import torch

from common import MODULES

RESIDUAL_SITES = tuple(n for n in MODULES if n.endswith(('attn.o_proj', 'mlp.down_proj')))


def site_layer(name):
    return int(name.split('.')[1])


def site_kind(name):
    return 'attn' if name.endswith('o_proj') else 'mlp'


class PairedMoments:
    """Streaming per-column correlation of two token series x[:, c], y[:, c]."""

    def __init__(self, count, device='cpu'):
        self.count = count
        self.n = 0
        self.offset = None                                       # (x0, y0), fixed by the first batch
        zeros = lambda: torch.zeros(count, dtype=torch.float64, device=device)
        self.sx, self.sy, self.sxx, self.syy, self.sxy = zeros(), zeros(), zeros(), zeros(), zeros()

    def update(self, x, y):
        """x, y: [tokens, count]. Reductions in FP32, totals in FP64."""
        assert x.shape == y.shape and x.shape[-1] == self.count, (x.shape, y.shape, self.count)
        x, y = x.detach().float(), y.detach().float()
        if self.offset is None:
            self.offset = (x.mean(0).double(), y.mean(0).double())
        x = x - self.offset[0].to(x.dtype)
        y = y - self.offset[1].to(y.dtype)
        self.n += x.shape[0]
        self.sx += x.sum(0).double()
        self.sy += y.sum(0).double()
        self.sxx += x.square().sum(0).double()
        self.syy += y.square().sum(0).double()
        self.sxy += (x*y).sum(0).double()
        return self

    def result(self):
        """kappa plus the bookkeeping that tells a reactive cancellation from a constant one."""
        assert self.n > 1, 'need at least two tokens'
        n = float(self.n)
        x0, y0 = self.offset
        mx, my = self.sx/n, self.sy/n
        vx = (self.sxx/n - mx.square()).clamp_min(0.)
        vy = (self.syy/n - my.square()).clamp_min(0.)
        cov = self.sxy/n - mx*my
        sx, sy = vx.sqrt(), vy.sqrt()
        kappa = cov/(sx*sy)                                      # NaN where either side is constant
        kappa = torch.where((sx > 0) & (sy > 0), kappa, torch.full_like(kappa, float('nan')))
        return dict(kappa=kappa, covariance=cov, count=self.n,
                    rho_mean=mx+x0, rho_std=sx, projection_mean=my+y0, projection_std=sy)


def _piece_map(pieces):
    if isinstance(pieces, dict):
        return pieces
    assert len(pieces) == len(MODULES), (len(pieces), len(MODULES))
    return dict(zip(MODULES, pieces))


@torch.no_grad()
def repair_correlations(model, pieces, rows_iter, sites=RESIDUAL_SITES, device='cuda', progress=None):
    """kappa_c for every piece of every residual-write site.

    model     the intact target, untouched: only observing forward hooks are attached and removed.
    pieces    FreePieces per module, either a {module name: FreePieces} map or a sequence aligned
              with common.MODULES. read is [d_in, C], write is [d_out, C].
    rows_iter iterable of token-id rows [batch, positions+1], exactly as common.tokens() stores
              them; the model is fed rows[:, :-1].
    sites     which residual-write modules to measure (default: all o_proj and down_proj).
    progress  optional callback(batches, tokens) after each batch.

    Returns dict(sites={name: result-of-PairedMoments plus later_write_norm_mean}, tokens=int,
                 batches=int, checks={...}).  Every tensor is [C] on `device`.
    """
    pieces = _piece_map(pieces)
    sites = tuple(sites)
    assert all(s in RESIDUAL_SITES for s in sites), 'only o_proj / down_proj write into the stream'
    n_layer = len(model.h)
    acc = {s: PairedMoments(pieces[s].count, device) for s in sites}
    later_norm = {s: torch.zeros((), dtype=torch.float64, device=device) for s in sites}
    block_in, block_out, site_in, site_out, final = {}, {}, {}, {}, {}
    checks, tokens, batches = {}, 0, 0
    hooks = []

    def pre(layer):
        def hook(_m, args):
            block_in[layer] = args[0]
        return hook

    def post(layer):
        def hook(_m, _args, out):
            block_out[layer] = out
        return hook

    def at_site(name):
        def hook(_m, args, out):
            site_in[name], site_out[name] = args[0], out
        return hook

    try:
        for layer in range(n_layer):
            hooks.append(model.h[layer].register_forward_pre_hook(pre(layer)))
            hooks.append(model.h[layer].register_forward_hook(post(layer)))
        # Every o_proj/down_proj is hooked, not just the requested sites: the additivity check below
        # needs both writes of a layer to prove where "immediately after this site" actually is.
        for name in RESIDUAL_SITES:
            hooks.append(model.get_submodule(name).register_forward_hook(at_site(name)))
        hooks.append(model.ln_f.register_forward_hook(lambda _m, args, _o: final.__setitem__('x', args[0])))

        for rows in rows_iter:
            ids = rows[:, :-1]
            with torch.autocast(device, dtype=torch.bfloat16):
                model(ids)
            assert len(final) == 1 and len(block_out) == n_layer, 'hooks did not fire'
            stream = final['x'].float().flatten(0, 1)                                  # [n, d_model]
            if not checks:
                # The block is x = x + attn(rms_1(x)); x = x + mlp(rms_2(x)), so these two identities
                # are what makes "the residual immediately after site s" well defined. Verify, do not
                # assume.
                worst_block, worst_final = 0., 0.
                for layer in range(n_layer):
                    got = (block_in[layer].float()
                           + site_out[f'h.{layer}.attn.o_proj'].float()
                           + site_out[f'h.{layer}.mlp.down_proj'].float())
                    ref = block_out[layer].float()
                    worst_block = max(worst_block, float((got-ref).norm()/ref.norm()))
                last = block_out[n_layer-1].float().flatten(0, 1)
                worst_final = float((last-stream).norm()/stream.norm())
                checks = dict(max_relative_block_additivity_error=worst_block,
                              relative_last_block_vs_ln_f_input_error=worst_final)
                assert worst_block < 1e-3 and worst_final < 1e-3, checks
            for name in sites:
                layer, kind = site_layer(name), site_kind(name)
                after = (block_in[layer].float() + site_out[name].float()) if kind == 'attn' else block_out[layer].float()
                later = stream - after.flatten(0, 1)                                   # [n, d_model]
                piece = pieces[name]
                rho = site_in[name].float().flatten(0, 1) @ piece.read.float()          # [n, C]
                projection = later @ piece.write.float()                                # [n, C]
                acc[name].update(rho, projection)
                later_norm[name] += later.norm(dim=-1).double().sum()
                del after, later, rho, projection
            tokens += ids.numel()
            batches += 1
            block_in.clear(); block_out.clear(); site_in.clear(); site_out.clear(); final.clear()
            del stream, ids, rows
            if progress is not None:
                progress(batches, tokens)
    finally:
        for hook in hooks:
            hook.remove()

    out = {}
    for name in sites:
        stats = acc[name].result()
        stats['later_write_norm_mean'] = float(later_norm[name]/tokens)
        out[name] = stats
    return dict(sites=out, tokens=tokens, batches=batches, checks=checks)
