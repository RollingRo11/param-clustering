"""Max-activating examples for every piece of a par-auto decomposition of the 67M target, on ~1B tokens of the Pile
stratified by sub-dataset. No labels.

  data     stream monology/pile-uncopyrighted, tokenize (GPT-NeoX), pack each sub-dataset's documents (EOS-separated) into
           its own 513-token rows (a row never mixes sub-datasets). Every sub-dataset is capped at an equal share of the
           target; the rare ones (Enron, NIH, PhilPapers, ...) cannot fill it within the scan budget and give what the
           scan found, and their shortfall goes to the others (water-filling). Rows are shuffled and written as uint16.
  harvest  one shard of the rows (rank::world): the piece score that ranks pieces in KLKeep and the circuits (Fisher:
           |r_c . h| * RMS over K=4 sampled next tokens of (g_k . u_c), as in training) at every position; per piece
           the TOP highest-scoring positions, the number of positions where it is among the token's top 128 pieces
           (overall and per sub-dataset) and the sum of its score. Checkpointed every SAVE_EVERY chunks; resumable.
  merge    combine the shards; write per piece its top examples with the token and a context window.
Usage:
  harvest67.py data --out DIR [--tokens 1e9]
  harvest67.py harvest --out DIR --checkpoint CKPT --rank R --world W
  harvest67.py merge --out DIR --checkpoint CKPT --world W
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('PAIR_ROOT', '/workspace/paired'))
if (ROOT/'code/compare').is_dir(): sys.path.insert(0, str(ROOT/'code/compare'))

import numpy as np

ROW, EOS = 513, 0
DATASET, TOKENIZER = 'monology/pile-uncopyrighted', 'EleutherAI/gpt-neox-20b'
TOP, MEMBER_K, CHUNK, SAVE_EVERY, FISHER_K = 20, 128, 16, 200, 4
WINDOW = (24, 8)                      # context tokens before and after the top position


def log(**fields): print(json.dumps(fields), flush=True)


# ---------------------------------------------------------------------------------------------------------- data
def make_data(args):
    from datasets import load_dataset
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(TOKENIZER)
    assert tok.token_to_id('<|endoftext|>') == EOS
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    target_rows = int(args.tokens) // (ROW - 1)
    n_subsets = 17                                        # pile-uncopyrighted's sub-datasets
    collect_cap = 2 * target_rows // n_subsets + 1       # rows kept per sub-dataset during the scan (room to redistribute)
    rows = collections.defaultdict(list)                 # subset -> list of [n, ROW] uint16 arrays
    count = collections.Counter(); pending = collections.defaultdict(list)
    buffer = collections.defaultdict(list)               # subset -> leftover tokens shorter than a row
    docs = collections.Counter(); scanned = 0; started = time.perf_counter()

    def flush(subset):
        texts = pending.pop(subset, [])
        if not texts: return
        ids = buffer[subset]
        for enc in tok.encode_batch(texts): ids.extend(enc.ids); ids.append(EOS)
        n = len(ids) // ROW
        if n:
            block = np.asarray(ids[:n * ROW], dtype=np.uint16).reshape(n, ROW)
            take = min(n, collect_cap - count[subset])
            if take > 0: rows[subset].append(block[:take]); count[subset] += take
            buffer[subset] = ids[n * ROW:] if count[subset] < collect_cap else []
        else: buffer[subset] = ids

    for doc in load_dataset(DATASET, split='train', streaming=True):
        scanned += 1
        subset = doc['meta']['pile_set_name']
        docs[subset] += 1
        if count[subset] < collect_cap:
            pending[subset].append(doc['text'])
            if len(pending[subset]) >= 256: flush(subset)
        if scanned % 200_000 == 0:
            log(scanned=scanned, minutes=round((time.perf_counter() - started) / 60, 1),
                rows={s: count[s] for s in sorted(count)}, docs_seen=dict(docs))
        full = len(count) >= n_subsets and all(count[s] >= collect_cap for s in count)
        if full or scanned >= args.max_docs: break
    for subset in list(pending): flush(subset)
    # Water-filling: equal shares, a sub-dataset short of its share gives everything, the rest is shared by the others.
    have = {s: count[s] for s in count}
    quota, left, open_ = {}, target_rows, sorted(have)
    while open_:
        share = left // len(open_)
        short = [s for s in open_ if have[s] <= share]
        if not short:
            for i, s in enumerate(open_): quota[s] = share + (1 if i < left - share * len(open_) else 0)
            break
        for s in short: quota[s] = have[s]; left -= have[s]; open_.remove(s)
    names = sorted(quota)
    data = np.concatenate([np.concatenate(rows[s])[:quota[s]] for s in names])
    labels = np.concatenate([np.full(quota[s], i, dtype=np.uint8) for i, s in enumerate(names)])
    order = np.random.default_rng(0).permutation(len(data))
    mm = np.lib.format.open_memmap(out/'rows.npy', mode='w+', dtype=np.uint16, shape=data.shape)
    mm[:] = data[order]; mm.flush(); del mm
    np.save(out/'subsets.npy', labels[order])
    manifest = dict(dataset=DATASET, tokenizer=TOKENIZER, row=ROW, rows=int(len(data)), tokens=int(len(data) * (ROW - 1)),
                    subsets=names, rows_per_subset={s: int(quota[s]) for s in names}, rows_available={s: int(have[s]) for s in names},
                    docs_scanned=scanned, docs_seen=dict(docs), shuffle_seed=0, minutes=round((time.perf_counter() - started) / 60, 1))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1))
    log(**manifest)


# ------------------------------------------------------------------------------------------------------- harvest
def harvest(args):
    import torch
    from common import MODULES, TARGET_SHA, credentials, target, sha
    from fisher_attribution import capture_fisher
    from train_pair import core, method
    credentials(); torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out)
    rows = np.load(out/'rows.npy', mmap_mode='r'); subsets = np.load(out/'subsets.npy')
    n_sub = int(subsets.max()) + 1
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['method'] == 'parauto' and saved['target_sha256'] == TARGET_SHA
    pieces = [core.FreePieces(p['read'].cuda(), p['write'].cuda()) for p in saved['factors']]
    total = sum(p.read.shape[1] for p in pieces)
    model = target()
    mine = np.arange(args.rank, len(rows), args.world)
    chunks = [mine[i:i + CHUNK] for i in range(0, len(mine), CHUNK)]
    path = out/f'state_{args.rank}_of_{args.world}.pt'
    if path.exists():
        state = torch.load(path, map_location='cuda', weights_only=False)
        assert state['checkpoint_sha256'] == sha(args.checkpoint)
    else:
        state = dict(done=0, checkpoint_sha256=sha(args.checkpoint), top_score=torch.full((TOP, total), -1., device='cuda'),
                     top_index=torch.full((TOP, total), -1, dtype=torch.long, device='cuda'),
                     member=torch.zeros(n_sub, total, dtype=torch.long, device='cuda'), score_sum=torch.zeros(total, dtype=torch.float64, device='cuda'),
                     positions=0)
    started, t0 = time.perf_counter(), time.perf_counter()
    for c in range(state['done'], len(chunks)):
        idx = chunks[c]
        batch = torch.from_numpy(np.asarray(rows[idx], dtype=np.int64)).cuda()
        cache = capture_fisher(model, batch, k=FISHER_K, seed=int(idx[0]))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            s = torch.cat([method.scores_of(e, p).float() for e, p in zip(cache, pieces)], -1)            # [positions, total]
        del cache
        with torch.no_grad():
            pos_index = (torch.from_numpy(idx).cuda()[:, None] * (ROW - 1) + torch.arange(ROW - 1, device='cuda')).flatten()
            v, i = s.topk(min(TOP, s.shape[0]), dim=0)                                                       # [TOP, total]
            vals = torch.cat([state['top_score'], v]); inds = torch.cat([state['top_index'], pos_index[i]])
            keep = vals.topk(TOP, dim=0).indices
            state['top_score'], state['top_index'] = vals.gather(0, keep), inds.gather(0, keep)
            members = s.topk(MEMBER_K, dim=1).indices                                                         # [positions, MEMBER_K]
            sub = torch.from_numpy(subsets[idx].astype(np.int64)).cuda().repeat_interleave(ROW - 1)
            state['member'].view(-1).index_add_(0, (sub[:, None] * total + members).flatten(),
                                                torch.ones(members.numel(), dtype=torch.long, device='cuda'))
            state['score_sum'] += s.sum(0, dtype=torch.float64); state['positions'] += s.shape[0]
        state['done'] = c + 1
        if state['done'] % SAVE_EVERY == 0 or state['done'] == len(chunks):
            tmp = path.with_suffix('.tmp'); torch.save(state, tmp); os.replace(tmp, path)
            rate = SAVE_EVERY * CHUNK * (ROW - 1) / (time.perf_counter() - t0); t0 = time.perf_counter()
            log(rank=args.rank, chunks=state['done'], of=len(chunks), tokens=state['positions'], tokens_per_s=round(rate),
                hours_left=round((len(chunks) - state['done']) * CHUNK * (ROW - 1) / rate / 3600, 2))
    log(rank=args.rank, done=True, tokens=state['positions'], minutes=round((time.perf_counter() - started) / 60, 1))


# --------------------------------------------------------------------------------------------------------- merge
def merge(args):
    import torch
    from tokenizers import Tokenizer
    from common import MODULES
    out = Path(args.out)
    manifest = json.loads((out/'manifest.json').read_text())
    states = [torch.load(out/f'state_{r}_of_{args.world}.pt', map_location='cpu', weights_only=False) for r in range(args.world)]
    assert len({s['checkpoint_sha256'] for s in states}) == 1
    vals = torch.cat([s['top_score'] for s in states]); inds = torch.cat([s['top_index'] for s in states])
    keep = vals.topk(TOP, dim=0).indices
    top_score, top_index = vals.gather(0, keep), inds.gather(0, keep)
    member = sum(s['member'] for s in states); score_sum = sum(s['score_sum'] for s in states)
    positions = sum(s['positions'] for s in states)
    torch.save(dict(top_score=top_score, top_index=top_index, member=member, score_sum=score_sum, positions=positions,
                    checkpoint_sha256=states[0]['checkpoint_sha256'], subsets=manifest['subsets']), out/'harvest.pt')
    rows = np.load(out/'rows.npy', mmap_mode='r'); subsets = np.load(out/'subsets.npy')
    tok = Tokenizer.from_pretrained(TOKENIZER)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    names = [f'{n}:{i}' for n, p in zip(MODULES, saved['factors']) for i in range(p['read'].shape[1])]
    assert len(names) == top_index.shape[1]
    with open(out/'examples.jsonl', 'w') as f:
        for piece in range(top_index.shape[1]):
            examples = []
            for score, flat in zip(top_score[:, piece].tolist(), top_index[:, piece].tolist()):
                if flat < 0: continue
                r, t = divmod(flat, ROW - 1)
                ids = rows[r]
                lo, hi = max(0, t - WINDOW[0]), min(ROW - 1, t + WINDOW[1] + 1)
                examples.append(dict(score=round(score, 5), row=int(r), position=int(t), subset=manifest['subsets'][int(subsets[r])],
                                     before=tok.decode(ids[lo:t].tolist()), token=tok.decode([int(ids[t])]),
                                     after=tok.decode(ids[t + 1:hi].tolist()), next_token=tok.decode([int(ids[t + 1])])))
            f.write(json.dumps(dict(piece=names[piece], index=piece,
                                    member_rate=float(member[:, piece].sum()) / positions,
                                    member_by_subset={s: int(member[i, piece]) for i, s in enumerate(manifest['subsets'])},
                                    mean_score=float(score_sum[piece]) / positions, examples=examples)) + '\n')
    log(merged=str(out/'harvest.pt'), examples=str(out/'examples.jsonl'), positions=positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['data', 'harvest', 'merge'])
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--tokens', type=float, default=1e9)
    ap.add_argument('--max-docs', type=int, default=8_000_000); ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--rank', type=int, default=0); ap.add_argument('--world', type=int, default=1)
    args = ap.parse_args()
    dict(data=make_data, harvest=harvest, merge=merge)[args.phase](args)


if __name__ == '__main__': main()
