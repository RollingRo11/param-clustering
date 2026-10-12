"""Circuits and the VPD paper's emoticon edit (Sec. 6) for one decomposition of the 67M target.

Emoticons: train rows 1,000,000-1,199,999 (never seen by the 5k-step decompositions, which used rows < 320,000), first
clean emoticon per row, up to 512 rows. Position t is an emoticon starter when token t is : ; = (optionally space-led),
token t+1 one of D P p o O ) ( -) -(, token t+2 starts with whitespace (or is past the row end), and token t is space-led
or token t-1 ends in whitespace or !?., (emoticon_tokens.json, GPT-NeoX tokenizer). 64 rows with no starter followed
by a continuation (from row 1,200,000 on) are the off-target text.

1. Circuit size at the emoticon starter. Every position keeps its own top k pieces (par-auto: Fisher K=8 score; VPD:
   causal importance), as KLKeep; KL(p_intact || p_masked) at the starter for k on a grid; per starter, the smallest k
   with KL <= 0.1 and <= 0.5. VPD also: its own explanation (mask = its labels, delta left out) and its size (label > 0.1).
2. Emoticon pieces. Share of starters at which each piece is in the top 128, against the share of the other positions
   of the same rows; the top pieces by the difference.
3. The edit. The most emoticon-selective piece among the residual-writing matrices (attn.o_proj, mlp.down_proj) gets
   its write vector replaced by s * alpha * u_o / ||u_o|| (u_o: the unembedding row of 'o'; s: the sign of the piece's
   mean activation at the starters), i.e. W' = W + (s alpha u_o/||u_o|| - u_c) r_c^T on the target model, as the
   paper's W' = W - (U_c + alpha u_o/||u_o||) V_c^T. For alpha on a grid: mean P(o) at the starters, the share where o
   is top-1, KL(original || edited) on the positions within 20 tokens of a starter (starter excluded) and on the
   off-target rows.
Usage: emoticon67.py --method parauto|vpd --checkpoint PATH --output JSON"""
import argparse
import copy
import json
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

ROWS = range(1_000_000, 1_200_000)
OFF_ROWS = 1_200_000
MAX_EMOTICONS, OFF_TARGET_ROWS, MICROBATCH, FISHER_K = 512, 64, 8, 8
K_GRID = (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096)
THRESHOLDS = (0.1, 0.5)
MEMBER_K = 128
ALPHAS = (1, 2, 4, 8, 16, 32, 64, 128)
SURROUND = 20


def find_emoticons(train, spec):
    V = spec['vocab']
    mask = lambda ids: np.isin(np.arange(V), ids)
    start, cont, after, before, spaced = (mask(spec[k]) for k in ('starters', 'continuations', 'after_ok', 'before_ok', 'spaced_starters'))
    hits = []
    for lo in range(ROWS.start, ROWS.stop, 20000):
        blk = np.array(train[lo:min(lo + 20000, ROWS.stop)])                    # [n, 513]
        t = np.arange(511)                                                      # starter positions with a next token in ids
        ok = start[blk[:, :511]] & cont[blk[:, 1:512]]
        ok &= after[blk[:, 2:513]]
        prev_ok = np.concatenate([np.zeros((len(blk), 1), bool), before[blk[:, :510]]], 1)
        ok &= spaced[blk[:, :511]] | prev_ok
        for i in np.nonzero(ok.any(1))[0]:
            hits.append((lo + int(i), int(t[ok[i]][0])))
            if len(hits) == MAX_EMOTICONS: return hits
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['parauto', 'vpd'], required=True)
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); credentials(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    from param_decomp.utils.distributed_utils import init_distributed
    init_distributed()
    started = time.perf_counter()
    vpd = args.method == 'vpd'
    spec = json.loads((Path(__file__).parent/'emoticon_tokens.json').read_text())
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
        factors = [(trainer.model.components[n].V.detach().float(), trainer.model.components[n].U.detach().T.float()) for n in MODULES]
    else:
        from fisher_attribution import capture_fisher
        pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
        factors = [(p.read.float(), p.write.float()) for p in pieces]
    del saved
    counts = [read.shape[1] for read, _ in factors]
    offsets = np.cumsum([0] + counts)
    total = sum(counts)
    train, manifest = tokens('train')
    hits = find_emoticons(train, spec)
    off_rows = []
    start_ids, cont_ids = set(spec['starters']), set(spec['continuations'])
    r = OFF_ROWS
    while len(off_rows) < OFF_TARGET_ROWS:
        row = np.array(train[r])
        if not any(int(a) in start_ids and int(b) in cont_ids for a, b in zip(row[:-1], row[1:])): off_rows.append(r)
        r += 1
    print(json.dumps(dict(emoticons=len(hits), off_target_rows=len(off_rows), seconds=round(time.perf_counter() - started))), flush=True)
    rows_of = lambda rs: torch.tensor(np.array([train[i] for i in rs], dtype=np.int64), device='cuda')

    def forward(ids, masks):
        if vpd:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                return trainer.model(ids, mask_infos=make_mask_infos(dict(zip(MODULES, masks)))).float()
        hooks = []
        try:
            for name, p, mask in zip(MODULES, pieces, masks):
                hooks.append(model.get_submodule(name).register_forward_hook(
                    lambda _layer, a, _o, p=p, mask=mask: ((a[0] @ p.read) * mask) @ p.write.T))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): return model(ids)[0].float()
        finally:
            for h in hooks: h.remove()

    def intact(ids):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): return model(ids)[0].float()

    def scores_and_inputs(rows, first):
        ids = rows[:, :-1]
        if vpd:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = trainer.model(ids, cache_type='input')
                ci = trainer.model.calc_causal_importances(base.cache, sampling='continuous').lower_leaky
                s = torch.cat([ci[n].float() for n in MODULES], -1).clamp(0., 1.)
                inputs = [base.cache[n].float() for n in MODULES]
            return s, inputs
        cache = capture_fisher(model, rows, k=FISHER_K, seed=first)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            s = torch.cat([method.scores_of(e, p).float().reshape(*ids.shape, -1) for e, p in zip(cache, pieces)], -1)
        inputs = [e['h'].float().view(*ids.shape, -1) for e in cache]
        return s, inputs

    # 1-2. Circuit sizes, emoticon pieces, and the pieces' activations at the starters (residual-writing matrices).
    writers = [i for i, n in enumerate(MODULES) if n.endswith(('attn.o_proj', 'mlp.down_proj'))]
    kl_grid = {k: [] for k in K_GRID}; own_kl, own_size = [], []
    member_e = torch.zeros(total, dtype=torch.float64, device='cuda'); member_o = torch.zeros_like(member_e); n_other = 0
    acts_e = {i: [] for i in writers}
    example = None
    for b in range(0, len(hits), MICROBATCH):
        chunk = hits[b:b + MICROBATCH]
        rows = rows_of([r for r, _ in chunk]); ids = rows[:, :-1]
        pos = torch.tensor([t for _, t in chunk], device='cuda'); at = torch.arange(len(chunk), device='cuda')
        s, inputs = scores_and_inputs(rows, b)
        logp = intact(ids)[at, pos].log_softmax(-1)
        p = logp.exp()
        order = s.argsort(dim=-1, descending=True)
        for k in K_GRID:
            mask = torch.zeros_like(s).scatter_(-1, order[..., :k], 1.)
            q = forward(ids, list(mask.split(counts, -1)))[at, pos].log_softmax(-1)
            kl_grid[k].append((p * (logp - q)).sum(-1).cpu())
        if vpd:
            q = forward(ids, list(s.split(counts, -1)))[at, pos].log_softmax(-1)
            own_kl.append((p * (logp - q)).sum(-1).cpu()); own_size.append((s[at, pos] > 0.1).sum(-1).float().cpu())
        top = order[..., :MEMBER_K]
        is_e = torch.zeros(ids.shape, dtype=torch.bool, device='cuda'); is_e[at, pos] = True
        member_e += torch.bincount(top[is_e].flatten(), minlength=total).double()
        member_o += torch.bincount(top[~is_e].flatten(), minlength=total).double(); n_other += int((~is_e).sum())
        for i in writers:
            acts_e[i].append((inputs[i][at, pos] @ factors[i][0]).cpu())
        if example is None:
            example = dict(row=chunk[0][0], position=chunk[0][1], text_ids=rows[0, max(0, chunk[0][1] - 30):chunk[0][1] + 3].tolist(),
                           order=order[0, chunk[0][1], :4096].tolist(), score=s[0, chunk[0][1], :].tolist() if vpd else None,
                           kl={k: float(kl_grid[k][-1][0]) for k in K_GRID})
        del s, inputs, order
        print(json.dumps(dict(emoticons_done=b + len(chunk), seconds=round(time.perf_counter() - started))), flush=True)
    kl_grid = {k: torch.cat(v) for k, v in kl_grid.items()}
    def smallest(tau):
        need = torch.full((len(hits),), float('inf'))
        for k in sorted(K_GRID, reverse=True): need[kl_grid[k] <= tau] = k
        return need
    circuit = {}
    for tau in THRESHOLDS:
        need = smallest(tau)
        reached = need[torch.isfinite(need)]
        circuit[str(tau)] = dict(reached_share=float(torch.isfinite(need).float().mean()),
                                 median=float(reached.median()) if len(reached) else None,
                                 quartiles=[float(reached.quantile(q)) for q in (.25, .75)] if len(reached) else None,
                                 per_emoticon=[None if not np.isfinite(x) else int(x) for x in need.tolist()])
    which = lambda c: (MODULES[int(np.searchsorted(offsets, c, side='right') - 1)], int(c - offsets[np.searchsorted(offsets, c, side='right') - 1]))
    share_e, share_o = member_e / len(hits), member_o / n_other
    selective = (share_e - share_o).argsort(descending=True)[:20].tolist()
    pieces_top = [dict(module=which(c)[0], index=which(c)[1], starter_share=float(share_e[c]), other_share=float(share_o[c])) for c in selective]
    writer_ids = torch.cat([torch.arange(int(offsets[i]), int(offsets[i + 1])) for i in writers]).cuda()
    best = int(writer_ids[(share_e - share_o)[writer_ids].argmax()])
    module, index = which(best)
    act = torch.cat(acts_e[MODULES.index(module)])[:, index]
    sign = 1. if float(act.mean()) >= 0 else -1.
    print(json.dumps(dict(circuit={t: {k: v for k, v in c.items() if k != 'per_emoticon'} for t, c in circuit.items()},
                          edit_piece=dict(module=module, index=index, starter_share=float(share_e[best]), other_share=float(share_o[best]),
                                          act_mean=float(act.mean()), act_abs_mean=float(act.abs().mean())),
                          top_pieces=pieces_top[:8])), flush=True)

    # 3. The edit on the target model's weight.
    layer = model.get_submodule(module)
    radford = type(layer).__name__ == 'RadfordConv1D'
    original = layer.weight.detach().clone()
    read, write = factors[MODULES.index(module)][0][:, index], factors[MODULES.index(module)][1][:, index]
    u_o = model.lm_head.weight[spec['o']].float(); u_o = u_o / u_o.norm()
    e_rows = rows_of([r for r, _ in hits]); off = rows_of(off_rows)
    e_pos = torch.tensor([t for _, t in hits], device='cuda')
    near = torch.zeros(e_rows[:, :-1].shape, dtype=torch.bool, device='cuda')
    for j, t in enumerate(e_pos.tolist()): near[j, max(0, t - SURROUND):t + SURROUND + 1] = True
    near[torch.arange(len(hits), device='cuda'), e_pos] = False
    reference = copy.deepcopy(model)                                            # the unedited target, for the KLs
    def unedited(ids):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): return reference(ids)[0].float().log_softmax(-1)

    def evaluate():
        p_o, top1, kl_near, n_near, kl_off, n_off = [], [], 0., 0, 0., 0
        for b in range(0, len(hits), MICROBATCH):
            base = unedited(e_rows[b:b + MICROBATCH, :-1]); q = intact(e_rows[b:b + MICROBATCH, :-1]).log_softmax(-1)
            at = torch.arange(q.shape[0], device='cuda'); pos = e_pos[b:b + MICROBATCH]
            p_o.append(q[at, pos, spec['o']].exp().cpu()); top1.append((q[at, pos].argmax(-1) == spec['o']).float().cpu())
            kl = (base.exp() * (base - q)).sum(-1)
            kl_near += float(kl[near[b:b + MICROBATCH]].sum()); n_near += int(near[b:b + MICROBATCH].sum())
        for b in range(0, len(off), MICROBATCH):
            base = unedited(off[b:b + MICROBATCH, :-1]); q = intact(off[b:b + MICROBATCH, :-1]).log_softmax(-1)
            kl = (base.exp() * (base - q)).sum(-1)
            kl_off += float(kl.sum()); n_off += kl.numel()
        return dict(p_o=float(torch.cat(p_o).mean()), o_top1=float(torch.cat(top1).mean()),
                    kl_surrounding=kl_near / n_near, kl_global=kl_off / n_off)

    edits = {'0': evaluate()}
    for alpha in ALPHAS:
        delta = torch.outer(sign * alpha * u_o - write, read)                    # [d_out, d_in]
        with torch.no_grad():
            layer.weight.copy_(original + (delta.T if radford else delta).to(original.dtype))
        edits[str(alpha)] = evaluate()
        print(json.dumps({f'alpha {alpha}': edits[str(alpha)]}), flush=True)
    with torch.no_grad(): layer.weight.copy_(original)
    save_json(dict(method=args.method, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), **meta,
                   data_sha256=manifest['train']['sha256'], rows=[ROWS.start, ROWS.stop], emoticons=[list(h) for h in hits],
                   off_target_rows=off_rows, ranking='CI' if vpd else f'Fisher K={FISHER_K}', k_grid=list(K_GRID),
                   kl_at_starter={str(k): dict(mean=float(v.mean()), median=float(v.median())) for k, v in kl_grid.items()},
                   circuit=circuit, own_explanation=dict(kl_mean=float(torch.cat(own_kl).mean()), kl_median=float(torch.cat(own_kl).median()),
                                                         size_median=float(torch.cat(own_size).median()), size_mean=float(torch.cat(own_size).mean()))
                   if vpd else None,
                   top_pieces=pieces_top, edit_piece=dict(module=module, index=index, starter_share=float(share_e[best]), other_share=float(share_o[best]),
                                                          sign=sign, act_mean=float(act.mean()), act_abs_mean=float(act.abs().mean()),
                                                          read_norm=float(read.norm()), write_norm=float(write.norm())),
                   edits=edits, example=example, piece_names=[[n, c] for n, c in zip(MODULES, counts)],
                   gpu=torch.cuda.get_device_name(), seconds=time.perf_counter() - started), args.output)
    print('EMOTICON_DONE', flush=True)


if __name__ == '__main__': main()
