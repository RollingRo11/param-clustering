"""Five robustness variants of Matryoshka par-auto, each one extra loss term on top of matryoshka67.py, which this
file imports and runs unchanged apart from its per-update accumulate (replaced here), batch (wrapped, to know the
update number) and, for prefix, checkpoint (wrapped, to save the keep rates beside each full checkpoint). Ideas from
the project note "Adversarial robustness ideas for Matryoshka" (13 Oct 2026); prefix from Rohan, 14 Oct 2026. Run with the
5k mass-10 settings (hard forward, Fisher, Lp 1, stratified k^-1.5 budgets one per context, mass penalty 10), so each
run differs from main_hard_lp1_strat_per1_pow1.5_mass10 only by its term.

MATRYOSHKA_ROBUST = energy | union | holes | attack | prefix, MATRYOSHKA_ROBUST_COEFF (default 1):
  energy  dropped-energy penalty. Per token and matrix: sum over the pieces the token drops of |r_c.h| ||u_c||, over
          ||W h||, divided per matrix by the same sum over all pieces for the initial factors (the first update's
          first 256 positions, fixed in robust_state.json). Mean over matrices and positions. No extra forward.
  union   borrowed explanations. Each context also keeps a donor token's hard keep set (a random position of the next
          context in the chunk, at that context's budget) at every one of its positions; KL of the union mask.
  holes   Matryoshka with holes. Keep the top k and every piece ranked below k + j, drop the band in between; j drawn
          from the budget density, k + j clipped to N - 1; KL of that mask. Straight-through gradients of both hard
          masks (sigmoid top-k at k and at k + j).
  attack  closed-form one-step attacker. G_c = dKL/dmask_c from the main backward at the hard keep set, summed over
          a context's positions where c is dropped. Every dropped piece with G_c > 0 is switched fully on at all of
          the context's positions (the maximiser of the linearised KL over sources in [0, 1]); KL of that mask.
  prefix  global-prefix Matryoshka. f_c, each piece's keep rate (share of the update's positions whose hard top-k set
          holds c), is an EMA over updates (decay PREFIX_DECAY, start 0 for every piece). From update PREFIX_WARMUP
          on: one n per update from the k^-1.5 budget density on [PREFIX_MIN, alive] (alive: pieces kept at least
          once in the previous update), G_n = the n pieces of highest f_c, and every position keeps its own hard top-k
          set plus G_n; KL of that mask. Straight-through gradient of the own set only; G_n gets none. Every
          MONITOR_EVERY updates it also logs the KL of G_n alone (no own set) on the monitor rows at PREFIX_CURVE.
union, holes, attack and prefix add coeff * (mean KL per context) like the main KL, as a second forward and backward after the
main one, with the scores recomputed (same values). Random draws (donor positions, band widths) come from a generator
seeded by the update number, so a resumed run makes the same draws."""
import json
import math
import os

import numpy as np
import torch

import matryoshka67 as mt
from common import CONTEXT, MODULES, save_json
from train_pair import core

VARIANT = os.environ.get('MATRYOSHKA_ROBUST', '')
assert VARIANT in ('energy', 'union', 'holes', 'attack', 'prefix'), VARIANT
COEFF = float(os.environ.get('MATRYOSHKA_ROBUST_COEFF', '1'))
assert COEFF > 0 and math.isfinite(COEFF)
assert mt.FORWARD == 'hard' and mt.ADV_COEFF == 0
REFERENCE = 256                      # positions for the energy normalisation
STATE = mt.OUT/'robust_state.json'
CONFIG = dict(variant=VARIANT, coeff=COEFF, note=__doc__.split('\n\n')[0])
PREFIX_DECAY = .99                   # keep-rate EMA: time constant ~100 updates
PREFIX_WARMUP = int(os.environ.get('MATRYOSHKA_PREFIX_WARMUP', '300'))   # updates before G_n is used
PREFIX_MIN = 10                      # smallest global budget n
PREFIX_CURVE = (10, 43, 129, 430, 1290, 4301, 12902)
PREFIX_FILE = 'robust_prefix.pt'     # keep rates, beside each checkpoint.pt
if VARIANT == 'prefix': CONFIG.update(decay=PREFIX_DECAY, warmup=PREFIX_WARMUP, n_min=PREFIX_MIN, curve=list(PREFIX_CURVE))

_step = None                         # the current update, set by batch()
_energy_denominators = None
_prefix = dict(rates=None, alive=None)   # prefix: keep-rate EMA [N] and the previous update's alive count
_monitor_rows = None
_original_batch = mt.batch
_original_checkpoint = mt.checkpoint


def batch(data, step):
    global _step
    _step = step
    return _original_batch(data, step)


def piece_sizes(entry, piece, weight, lo, hi):
    """|r_c.h| ||u_c|| / ||W h|| for positions lo:hi: the size of each piece's write, as a share of the matrix's. [n, C]"""
    h = entry['h'][lo:hi]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        read = h @ piece.read
    size = read.float().abs() * piece.write.float().pow(2).sum(0).add(1e-30).sqrt()
    with torch.no_grad():
        whole = (h.float() @ weight.float().T).norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return size / whole


def start(cache, leaves, weights, lo):
    """First update of a fresh run: fix the energy denominators and write robust_state.json."""
    global _energy_denominators
    state = dict(CONFIG)
    if VARIANT == 'energy':
        with torch.no_grad():
            _energy_denominators = [float(piece_sizes(e, l, w, lo, lo + REFERENCE).sum(-1).mean())
                                    for e, l, w in zip(cache, leaves, weights)]
        state['energy_denominators'] = dict(zip(MODULES, _energy_denominators))
    save_json(state, STATE)
    print(json.dumps(dict(robust_state=state)), flush=True)


def resume():
    """A resumed run must have the same variant and coefficient, and reuses the saved denominators (energy) or the
    keep rates saved beside the latest full checkpoint (prefix)."""
    global _energy_denominators
    if not STATE.exists(): return
    state = json.loads(STATE.read_text())
    assert {k: state[k] for k in CONFIG if k != 'note'} == {k: v for k, v in CONFIG.items() if k != 'note'}, (state, CONFIG)
    if VARIANT == 'energy': _energy_denominators = [state['energy_denominators'][n] for n in MODULES]
    if VARIANT == 'prefix' and (mt.OUT/'latest_checkpoint.json').exists():
        latest = json.loads((mt.OUT/'latest_checkpoint.json').read_text())
        saved = torch.load(mt.Path(latest['full_checkpoint']).parent/PREFIX_FILE, map_location='cuda')
        assert saved['completed'] == latest['step'], (saved['completed'], latest)
        _prefix.update(rates=saved['rates'], alive=saved['alive'])


def checkpoint(value, path):
    """matryoshka67's checkpoint, plus (prefix) the keep rates beside each full checkpoint, at the same update."""
    _original_checkpoint(value, path)
    if VARIANT == 'prefix' and mt.Path(path).name == 'checkpoint.pt':
        _original_checkpoint(dict(completed=value['completed'], rates=_prefix['rates'], alive=_prefix['alive']),
                             mt.Path(path).parent/PREFIX_FILE)


def prefix_set(n):
    """G_n as a 0/1 vector [N]: the n pieces of highest keep rate."""
    g = torch.zeros_like(_prefix['rates'])
    return g.scatter_(0, _prefix['rates'].topk(n, sorted=False).indices, 1.)


def prefix_alone_curve(model, pieces):
    """KL of G_n alone (no own set) on the monitor rows, at each n of PREFIX_CURVE below the piece count."""
    global _monitor_rows
    if _monitor_rows is None:
        val, _ = mt.tokens('val')
        _monitor_rows = torch.from_numpy(np.array(val[mt.MONITOR_ROWS], dtype=np.int64)).cuda()
    ids = _monitor_rows[:, :-1]
    frozen = [core.FreePieces(p.read.detach(), p.write.detach()) for p in pieces]
    curve = {}
    with torch.no_grad():
        target = mt.intact_logp(model, ids)
        for n in PREFIX_CURVE:
            if n >= len(_prefix['rates']): continue
            alphas = [a.expand(*ids.shape, -1) for a in prefix_set(n).split(mt.COUNTS)]
            curve[n] = float(mt.kl_per_position(target, mt.masked_logits(model, ids, frozen, alphas)).mean())
    return curve


def energy_term(cache, leaves, weights, mask, lo, hi, groups):
    drop = 1 - mask.detach()
    terms, first = [], 0
    for e, l, w, count, d in zip(cache, leaves, weights, mt.COUNTS, _energy_denominators):
        terms.append((piece_sizes(e, l, w, lo, hi) * drop[:, first:first+count]).sum(-1) / d)
        first += count
    return torch.stack(terms).mean(0).view(groups, -1).mean(-1)          # per budget group


def union_mask(mask, shape, generator):
    contexts, positions = shape
    assert contexts > 1, 'union needs at least two contexts per chunk'
    hard = mask.detach().view(contexts, positions, -1)
    picks = torch.randint(positions, (contexts,), generator=generator).to(mask.device)
    donors = (torch.arange(contexts, device=mask.device) + 1) % contexts
    donor = hard[donors, picks]                                         # [contexts, N]
    return mask + ((1 - hard) * donor[:, None, :]).flatten(0, 1)


def hole_mask(mask, s, ks, group, generator, total):
    widths = [mt.budget_at(float(torch.rand((), generator=generator, dtype=torch.float64)), total) for _ in ks]
    outer = [min(total - 1, k + j) for k, j in zip(ks, widths)]
    outer_mask, _ = mt.forward_mask(s, outer, 'hard', group)
    return mask + (1 - outer_mask), widths


def attack_mask(mask, gradient, shape):
    drop = 1 - mask.detach().view(*shape, -1)
    gain = (gradient.view(*shape, -1) * drop).sum(1)                    # [contexts, N]: dKL/dsource, shared source
    on = (gain > 0).to(mask.dtype)
    return mask + (drop * on[:, None, :]).flatten(0, 1), on


def prefix_budget(u, hi):
    """The global budget n at quantile u of the k budgets' density (k^-BUDGET_POWER), on [PREFIX_MIN, hi]."""
    lo, hi = PREFIX_MIN, max(hi, PREFIX_MIN)
    if mt.BUDGET_POWER == 1:
        n = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        e = 1 - mt.BUDGET_POWER
        n = (lo ** e + u * (hi ** e - lo ** e)) ** (1 / e)
    return min(hi, max(1, round(n)))


def masked_kl(model, ids, target, leaves, mask, groups):
    alphas = [a.reshape(*ids.shape, -1) for a in mask.split(mt.COUNTS, -1)]
    return mt.kl_per_position(target, mt.masked_logits(model, ids, leaves, alphas)).view(groups, -1).mean(-1)


def accumulate(model, rows, cache, feasible, draws, forward=mt.FORWARD, lp=None, target_logp=None, chunk=None, adversary=None):
    """matryoshka67.accumulate (without its adversary) plus the variant's term; same returns."""
    assert adversary is None and forward == 'hard' and _step is not None
    chunk = chunk or mt.CHUNK
    per = len(rows) // len(draws)
    assert per * len(draws) == len(rows) and chunk % per == 0, (len(rows), len(draws), chunk)
    leaves = [core.FreePieces(f.read.detach().requires_grad_(), f.write.detach().requires_grad_()) for f in feasible]
    weights = [model.get_submodule(n).weight for n in MODULES]
    total = sum(mt.COUNTS)
    count = len(draws)
    generator = torch.Generator().manual_seed(7_000_003 + _step)
    pending, extras, side = [], [], []
    active = False
    if VARIANT == 'prefix':
        if _prefix['rates'] is None: _prefix['rates'] = torch.zeros(total, device='cuda')
        active = _step >= PREFIX_WARMUP and _prefix['alive'] is not None
        info = dict(alive=_prefix['alive'])
        if active:
            if _step % mt.MONITOR_EVERY == 0: info['curve'] = prefix_alone_curve(model, feasible)
            info['n'] = prefix_budget(float(torch.rand((), generator=generator, dtype=torch.float64)), _prefix['alive'])
            g = prefix_set(info['n'])
        kept, positions = torch.zeros(total, device='cuda'), 0
        side.append(info)
    for first in range(0, len(rows), chunk):
        block = rows[first:first+chunk]
        ids = block[:, :-1]
        lo, hi = first*CONTEXT, (first+len(block))*CONTEXT
        ks = draws[first//per:(first+len(block))//per]
        if not STATE.exists(): start(cache, leaves, weights, lo)
        target = mt.intact_logp(model, ids) if target_logp is None else target_logp[lo:hi].view(*ids.shape, -1)
        parts = [] if lp else None
        s = mt.log_scores(cache, leaves, lo, hi, parts_out=parts)
        mask, soft = mt.forward_mask(s, ks, forward, per*CONTEXT)
        captured = []
        if VARIANT == 'attack': mask.register_hook(lambda g, store=captured: store.append(g.detach()))
        kl = masked_kl(model, ids, target, leaves, mask, len(ks))
        term = mt.lp_term(parts, lp) if lp else None
        loss = kl.sum() if term is None else kl.sum() + lp['coeff'] * term * len(ks)
        extra = None
        if VARIANT == 'energy':
            extra = energy_term(cache, leaves, weights, mask, lo, hi, len(ks))
            loss = loss + COEFF * extra.sum()
        (loss / count).backward()
        if VARIANT == 'prefix':
            kept += mask.detach().sum(0); positions += mask.shape[0]
        if VARIANT in ('union', 'holes', 'attack') or active:
            s2 = mt.log_scores(cache, leaves, lo, hi)
            mask2, _ = mt.forward_mask(s2, ks, forward, per*CONTEXT)
            if VARIANT == 'union':
                varied = union_mask(mask2, ids.shape, generator)
            elif VARIANT == 'holes':
                varied, widths = hole_mask(mask2, s2, ks, per*CONTEXT, generator, total)
                side.extend(widths)
            elif VARIANT == 'attack':
                varied, on = attack_mask(mask2, captured[0], ids.shape)
                side.append(on.mean(-1).detach())
            else:
                varied = mask2 + (1 - mask2.detach()) * g      # own hard top-k set plus G_n, same G_n at every position
            extra = masked_kl(model, ids, target, leaves, varied, len(ks))
            (COEFF * extra.sum() / count).backward()
            del s2, mask2, varied
        if extra is not None: extras.append(extra.detach())
        with torch.no_grad():
            on_soft = soft > .5
            pending.append((ks, kl.detach(), on_soft.sum(-1).float().view(len(ks), -1).mean(-1),
                            (soft * ~on_soft).sum(-1).view(len(ks), -1).mean(-1), None if term is None else term.detach()))
        del target, s, mask, soft, kl, parts, term, loss, extra, captured
    if VARIANT == 'prefix':
        with torch.no_grad():
            _prefix['rates'].mul_(PREFIX_DECAY).add_(kept / positions, alpha=1 - PREFIX_DECAY)
        _prefix['alive'] = int((kept > 0).sum())
    stats = []
    for ks, kl, on, leak, term in pending:
        kl, on, leak = kl.tolist(), on.tolist(), leak.tolist()
        term = None if term is None else float(term)
        for j, k in enumerate(ks):
            stats.append(dict(k=k, kl=kl[j], on_over_k=on[j] / k, soft_leak=leak[j] / k))
            if term is not None: stats[-1]['lp_normalized'] = term
    values = torch.cat(extras).tolist() if extras else []
    for row, v in zip(stats, values): row[f'robust_{VARIANT}'] = v
    log(stats, values, side, total)
    return leaves, stats


def log(stats, values, side, total):
    key = 'energy' if VARIANT == 'energy' else f'{VARIANT}_kl'
    row = {f'train/robust_{key}': float(np.mean(values))} if values else {}
    for lo, hi, label in ((0, 100, 'lt100'), (100, 1000, '100_1000'), (1000, 10000, '1000_10000'), (10000, total, 'ge10000')):
        inside = [v for s, v in zip(stats, values) if lo <= s['k'] < hi]
        if inside: row[f'train/robust_{key}_k_{label}'] = float(np.mean(inside))
    if VARIANT == 'holes': row['train/robust_hole_width_mean'] = float(np.mean(side))
    if VARIANT == 'attack': row['train/robust_attack_on_share'] = float(torch.cat(side).mean())
    if VARIANT == 'prefix':
        info, = side
        row['train/robust_prefix_alive_next'] = _prefix['alive']
        row['train/robust_prefix_rate_max'] = float(_prefix['rates'].max())
        if 'n' in info: row['train/robust_prefix_n'] = info['n']
        for n, v in info.get('curve', {}).items(): row[f'train/robust_prefix_alone_kl_n{n}'] = v
    if not all(math.isfinite(v) for v in row.values()): raise RuntimeError(f'Nonfinite robust metrics: {row}')
    import wandb
    if wandb.run is not None: wandb.run.log({'update': _step + 1, **row})
    if (_step + 1) % 10 == 0 or _step < 5:
        print(json.dumps(dict(update=_step + 1, **{k.split('/', 1)[1]: v for k, v in row.items()},
                              robust_by_budget=values)), flush=True)


if __name__ == '__main__':
    resume()
    mt.batch, mt.accumulate, mt.checkpoint = batch, accumulate, checkpoint
    mt.main()
