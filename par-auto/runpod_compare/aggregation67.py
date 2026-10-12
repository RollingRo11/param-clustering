"""Explanation aggregation, after the audit "Do VPD's explanations aggregate?" (LessWrong, 2026; code
github.com/angsten/vpd-audit), on our held-out rows: recipients val rows 0-1023, donors val rows 1024-2047 (disjoint),
512 positions each.

A token's explanation. VPD: its causal-importance labels g in [0, 1] (lower_leaky, clamped), computed once on the
unmasked text; the explanation's mask is g itself with the delta component left out; 'needed' means g > 0.1 (the
paper's and the audit's cut-off). par-auto: the top-k pieces by the Fisher K=8 score (the KLKeep ranking), mask 1 on
them and 0 elsewhere; 'needed' means in the top k.
Aggregating n donor tokens: the union U of the needed sets of the first n positions of a seeded order of all donor
positions (8 draws; the sets nest). Text rungs take whole donor texts in a seeded text order (1 text, 64 texts, all).
The mask at every recipient token is 1 on U and the recipient's own explanation elsewhere; one U per draw is used for
every recipient text.
Harm: KL(P_intact || Q_masked) per position (bf16 logits, fp32 KL), averaged over a text's 512 positions and then over
texts; reported as the rise over the recipient's own explanation, with 95% bootstrap intervals over texts (draws
averaged per text first, so the draws are held fixed). Also the share of positions whose top-1 prediction changes and
the next-token accuracy.
Controls on the token rungs, with the same draws: uniform, |U| pieces drawn uniformly from the alive set (VPD: mean label
over the donor positions > 1e-6; par-auto: needed at >= 1 donor position), nested; frequency-matched, exponential clocks
with rate -ln(1 - u_c), u_c the share of donor positions that need c: a piece is in the n-token set if its clock rings
by n.
Usage: aggregation67.py --method parauto|vpd --checkpoint PATH --output JSON [--budgets 169,430] [--smoke]"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np
import torch

from common import MODULES, TARGET_SHA, credentials, target, tokens, save_json, sha
from train_pair import core, method

POSITIONS = 512
RECIPIENTS, DONORS = range(0, 1024), range(1024, 2048)
TOKEN_RUNGS = (1, 2, 4, 8, 16, 32, 64)
TEXT_RUNGS = (1, 64)               # plus all donor texts
DRAWS = 8
NEEDED = 0.1                       # VPD: label above this = needed
ALIVE = 1e-6                       # VPD: alive if its mean label over the donor positions is above this
FISHER_K = 8


def pack(needed):
    """bool [m, N] -> uint8 [m, N/8]"""
    shifts = torch.arange(7, -1, -1, device=needed.device, dtype=torch.int32)
    return (needed.view(needed.shape[0], -1, 8).to(torch.int32) << shifts).sum(-1).to(torch.uint8)


def unpack(packed):
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> shifts) & 1).bool().view(packed.shape[0], -1)


def bootstrap(values, generator, resamples=10000):
    """Mean of a per-text array and the 95% interval of that mean, resampling texts."""
    v = torch.as_tensor(values, dtype=torch.float64)
    means = v[torch.randint(len(v), (resamples, len(v)), generator=generator)].mean(-1)
    return dict(mean=float(v.mean()), lo=float(means.quantile(.025)), hi=float(means.quantile(.975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['parauto', 'vpd'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--budgets', default='169,430', help='par-auto explanation sizes, pieces per token')
    ap.add_argument('--chunk', type=int, default=16, help='texts per forward')
    ap.add_argument('--smoke', action='store_true', help='32 recipients and 32 donors')
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    started = time.perf_counter()
    vpd = args.method == 'vpd'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == args.method and saved['target_sha256'] == TARGET_SHA
    meta = dict(completed=saved.get('completed'), training_positions=saved.get('training_positions'))
    model = target()
    if vpd:
        from train_pair import VPD
        from param_decomp.models.component_model import handle_deprecated_state_dict_keys_
        from param_decomp.models.components import make_mask_infos
        trainer = VPD(model, evaluation=True); handle_deprecated_state_dict_keys_(saved['factors'])
        trainer.model.load_state_dict(saved['factors']); trainer.model.eval().requires_grad_(False)
        counts = [trainer.model.module_to_c[n] for n in MODULES]
    else:
        from fisher_attribution import capture_fisher
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
        counts = [p.read.shape[1] for p in pieces]
    del saved
    total = sum(counts)
    assert total % 8 == 0
    budgets = [int(b) for b in args.budgets.split(',')] if not vpd else [None]
    data, manifest = tokens('val')
    assert len(data) >= DONORS.stop
    recipients, donors = (range(0, 32), range(1024, 1056)) if args.smoke else (RECIPIENTS, DONORS)
    text_rungs = [m for m in TEXT_RUNGS if m < len(donors)] + [len(donors)]

    def rows_at(first, n):
        return torch.tensor(np.array(data[first:first+n], dtype=np.int64), device='cuda')

    def explain(rows, first):
        """par-auto: Fisher K=8 scores; VPD: labels in [0, 1]; [n, 512, total] fp32."""
        ids = rows[:, :-1]
        if not vpd:
            cache = capture_fisher(model, rows, k=FISHER_K, seed=first)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = torch.cat([method.scores_of(e, p).reshape(*ids.shape, -1) for e, p in zip(cache, pieces)], -1).float()
            del cache
            return out
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            base = trainer.model(ids, cache_type='input')
            ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
            out = torch.cat([ci[n] for n in MODULES], -1).float().clamp(0., 1.)
        del base, ci
        return out

    def forward(ids, mask):
        """Logits with every matrix replaced by its masked pieces (VPD: masked components, delta left out)."""
        parts = mask.split(counts, -1)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if vpd: return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, parts)))).float()
            hooks = []
            try:
                for name, p, m in zip(MODULES, pieces, parts):
                    hooks.append(model.get_submodule(name).register_forward_hook(
                        lambda _layer, inputs, _out, p=p, m=m: ((inputs[0] @ p.read) * m) @ p.write.T))
                return model(ids)[0].float()
            finally:
                for h in hooks: h.remove()

    # 1. Donors: each position's needed set (par-auto: its top max(budgets) pieces in order; VPD: packed label > 0.1).
    width = max(budgets) if not vpd else None
    store, label_sum = [], torch.zeros(total, dtype=torch.float64, device='cuda')
    usage = {b: torch.zeros(total, dtype=torch.float64, device='cuda') for b in budgets}
    for first in range(donors.start, donors.stop, args.chunk):
        n = min(args.chunk, donors.stop - first)
        e = explain(rows_at(first, n), first).flatten(0, 1)
        if vpd:
            needed = e > NEEDED
            store.append(pack(needed)); label_sum += e.double().sum(0); usage[None] += needed.double().sum(0)
            del needed
        else:
            top = e.topk(width, dim=-1, sorted=True).indices.to(torch.int32)          # most important first
            store.append(top)
            for b in budgets: usage[b] += torch.bincount(top[:, :b].flatten().long(), minlength=total).double()
        del e
    store = torch.cat(store)
    positions = store.shape[0]
    assert positions == len(donors) * POSITIONS
    alive = {b: (label_sum / positions > ALIVE) if vpd else (usage[b] > 0) for b in budgets}
    print(dict(donor_positions=positions, alive={str(b): int(a.sum()) for b, a in alive.items()},
               seconds=round(time.perf_counter() - started)), flush=True)

    def union(index, b):
        """Pieces needed at any of these donor positions: bool [total]."""
        out = torch.zeros(total, dtype=torch.bool, device='cuda')
        for i in range(0, len(index), 4096):
            rows = store[index[i:i+4096].cuda()]
            if vpd: out |= unpack(rows).any(0)
            else: out[rows[:, :b].flatten().long()] = True
        return out

    # 2. The aggregated sets, per draw: real unions, and on the token rungs the two controls.
    configs = {b: [] for b in budgets}
    for d in range(DRAWS):
        g = torch.Generator().manual_seed(9000 + d)
        order, text_order = torch.randperm(positions, generator=g), torch.randperm(len(donors), generator=g)
        exponential = torch.empty(total, dtype=torch.float64).exponential_(generator=g)
        piece_order = torch.randperm(total, generator=g)
        for b in budgets:
            u = (usage[b] / positions).cpu()
            clock = torch.where(u > 0, exponential / -torch.log1p(-u), torch.full_like(u, float('inf')))   # rate -ln(1 - u)
            alive_order = piece_order[alive[b].cpu()[piece_order]]
            for n in TOKEN_RUNGS:
                real = union(order[:n], b)
                uniform = torch.zeros(total, dtype=torch.bool); uniform[alive_order[:int(real.sum())]] = True
                for kind, chosen in (('real', real), ('uniform', uniform.cuda()), ('frequency', (clock <= n).cuda())):
                    configs[b].append(dict(kind=kind, rung=f'{n} tokens', tokens=n, draw=d, union=chosen, size=int(chosen.sum())))
            for m in text_rungs:
                if m == len(donors) and d: continue          # all donor texts: one union for every draw
                index = (text_order[:m, None] * POSITIONS + torch.arange(POSITIONS)).flatten()
                chosen = union(index, b)
                configs[b].append(dict(kind='real', rung=f'{m} texts', tokens=m * POSITIONS, draw=d, union=chosen, size=int(chosen.sum())))
    del store
    print(dict(configs={str(b): len(c) for b, c in configs.items()}, seconds=round(time.perf_counter() - started)), flush=True)

    # 3. Recipients: own explanation, then every aggregated set, per text.
    record = {b: dict(own=[[], [], []], cfg=[[[], [], []] for _ in configs[b]]) for b in budgets}
    intact_accuracy, own_size, sanity = [], [], {}
    for first in range(recipients.start, recipients.stop, args.chunk):
        tick = time.perf_counter()
        n = min(args.chunk, recipients.stop - first)
        rows = rows_at(first, n); ids, nxt = rows[:, :-1], rows[:, 1:]
        e = explain(rows, first)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16): intact = model(ids)[0]
            logp = intact.float().log_softmax(-1); p = logp.exp(); top1 = logp.argmax(-1)
            del intact
            intact_accuracy.append((top1 == nxt).float().mean(-1).cpu())

            def score(mask):
                q = forward(ids, mask).log_softmax(-1)
                pred = q.argmax(-1)
                return (p * (logp - q)).sum(-1).mean(-1).cpu(), (pred != top1).float().mean(-1).cpu(), (pred == nxt).float().mean(-1).cpu()

            if vpd: own_size.append((e > NEEDED).float().sum(-1).mean(-1).cpu())
            if first == recipients.start:      # every piece on: the decomposition itself (par-auto exact; VPD without its delta)
                sanity['all_on_kl'] = float(score(torch.ones_like(e))[0].mean())
            for b in budgets:
                own = e if vpd else torch.zeros_like(e).scatter_(-1, e.topk(b, dim=-1, sorted=False).indices, 1.)
                for slot, value in zip(record[b]['own'], score(own)): slot.append(value)
                for c, slots in zip(configs[b], record[b]['cfg']):
                    for slot, value in zip(slots, score(torch.where(c['union'], 1., own))): slot.append(value)
                del own
        del e, logp, p
        print(dict(recipients_done=first + n - recipients.start, chunk_seconds=round(time.perf_counter() - tick, 1),
                   own_kl={str(b): round(float(torch.cat(record[b]['own'][0]).mean()), 4) for b in budgets}), flush=True)

    # 4. Summary: per kind and rung, draws averaged per text, bootstrap over texts.
    generator = torch.Generator().manual_seed(0)
    summary, per_text = {}, {}
    for b in budgets:
        own = [torch.cat(s) for s in record[b]['own']]
        cfg = [[torch.cat(s) for s in slots] for slots in record[b]['cfg']]
        groups = {}
        for i, c in enumerate(configs[b]): groups.setdefault((c['kind'], c['rung'], c['tokens']), []).append(i)
        rows = []
        for (kind, rung, n_tokens), index in groups.items():
            kl = torch.stack([cfg[i][0] for i in index]).mean(0)
            rows.append(dict(kind=kind, rung=rung, donor_tokens=n_tokens, draws=len(index),
                             pieces=float(np.mean([configs[b][i]['size'] for i in index])), pieces_per_draw=[configs[b][i]['size'] for i in index],
                             kl=float(kl.mean()), rise=bootstrap(kl - own[0], generator),
                             top1_changed=float(torch.stack([cfg[i][1] for i in index]).mean()),
                             next_token_accuracy=float(torch.stack([cfg[i][2] for i in index]).mean())))
        key = 'vpd_labels' if vpd else f'top{b}'
        summary[key] = dict(explanation_size=float(torch.cat(own_size).mean()) if vpd else b, alive=int(alive[b].sum()),
                            own_kl=bootstrap(own[0], generator), own_top1_changed=float(own[1].mean()),
                            own_next_token_accuracy=float(own[2].mean()), rows=rows)
        per_text[key] = dict(own_kl=own[0].tolist(), kl={f"{c['kind']}|{c['rung']}|{c['draw']}": cfg[i][0].tolist() for i, c in enumerate(configs[b])})
    result = dict(method=args.method, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta,
                  recipients=[recipients.start, recipients.stop], donors=[donors.start, donors.stop], positions_per_text=POSITIONS,
                  data_sha256=manifest['val']['sha256'], ranking='Fisher K=8 top-k' if not vpd else 'VPD causal importance (lower_leaky)',
                  needed='top k' if not vpd else f'label > {NEEDED}', draws=DRAWS, token_rungs=list(TOKEN_RUNGS), text_rungs=text_rungs,
                  intact_next_token_accuracy=float(torch.cat(intact_accuracy).mean()), sanity=sanity, smoke=args.smoke,
                  gpu=torch.cuda.get_device_name(), seconds=time.perf_counter() - started, results=summary)
    save_json(result, args.output)
    save_json(per_text, args.output.with_name(args.output.stem + '_per_text.json'))
    for key, s in summary.items():
        print(key, dict(own_kl=round(s['own_kl']['mean'], 4), size=s['explanation_size']), flush=True)
        for r in s['rows']: print('  ', r['kind'], r['rung'], round(r['pieces']), round(r['rise']['mean'], 4), flush=True)
    print('AGGREGATION_DONE', flush=True)


if __name__ == '__main__': main()
