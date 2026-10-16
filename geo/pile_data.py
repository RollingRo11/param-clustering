"""Stratified sampling of monology/pile-uncopyrighted by meta.pile_set_name.

Streaming the dataset and taking documents in arrival order is badly skewed:
over the first 3,000 documents it is 29.9% Pile-CC, 17.2% PubMed Abstracts,
17.1% StackExchange, and 0.03% Ubuntu IRC. A pilot of a few hundred documents
is therefore close to "Pile-CC plus whatever else arrived first", which is a
poor basis for a decomposition meant to cover diverse text.

This builds an equal-quota sample instead: every subset contributes the same
number of `seq`-token blocks, each block drawn from that subset's documents
only (so a block is never a splice of Github and PubMed). Blocks are then
shuffled deterministically, so the held-out tail is as diverse as the pilot.

Subsets that cannot fill their quota within the scan budget are reported and
their shortfall is redistributed over the subsets that can.

Results are cached under /dev/shm keyed by the request, because a stratified
scan reads far more documents than a sequential one and every sensor/seed in a
sweep would otherwise repeat it.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import torch

DATASET = "monology/pile-uncopyrighted"
CACHE = Path("/dev/shm/geo67_piledata")


def load_pile_blocks(tok, n_blocks, seq, seed=0, max_docs=400_000,
                     tokenizer_name="", verbose=True):
    """Return (IDS [n_blocks, seq] int64 cpu, labels list[str], stats dict)."""
    key = hashlib.sha1(json.dumps(
        [DATASET, tokenizer_name, n_blocks, seq, seed, max_docs]).encode()
    ).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"strat_{key}.pt"
    if f.exists():
        d = torch.load(f, map_location="cpu", weights_only=False)
        if verbose:
            print(f"[pile] cache hit {f.name}: {tuple(d['ids'].shape)}, "
                  f"{len(set(d['labels']))} subsets", flush=True)
        return d["ids"], d["labels"], d["stats"]

    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True)

    # Phase 1: bucket tokens per subset until every subset has enough, or the
    # scan budget runs out. Quota is revised upward as new subsets appear.
    buf = collections.defaultdict(list)      # subset -> pending token ids
    blocks = collections.defaultdict(list)   # subset -> list of seq-token lists
    n_docs = 0
    for r in ds:
        n_docs += 1
        s = r["meta"]["pile_set_name"]
        b = buf[s]
        b.extend(tok.encode(r["text"]))
        while len(b) >= seq:
            blocks[s].append(b[:seq])
            del b[:seq]
        quota = max(1, n_blocks // max(1, len(blocks)))
        if n_docs >= max_docs:
            break
        if len(blocks) >= 8 and all(len(v) >= quota for v in blocks.values()) \
                and sum(len(v) for v in blocks.values()) >= n_blocks:
            break

    sets = sorted(blocks)
    assert sets, "no documents read"
    # Phase 2: equal quota, redistributing the shortfall of thin subsets.
    want = {s: n_blocks // len(sets) for s in sets}
    for i in range(n_blocks - sum(want.values())):
        want[sets[i % len(sets)]] += 1
    short = {s: max(0, want[s] - len(blocks[s])) for s in sets}
    spare = sum(short.values())
    for s in sets:
        want[s] = min(want[s], len(blocks[s]))
    while spare > 0:
        grew = False
        for s in sets:
            if spare <= 0:
                break
            if len(blocks[s]) > want[s]:
                want[s] += 1
                spare -= 1
                grew = True
        if not grew:
            break

    out, labels = [], []
    for s in sets:
        for blk in blocks[s][:want[s]]:
            out.append(blk)
            labels.append(s)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(out), generator=g).tolist()
    out = [out[i] for i in perm]
    labels = [labels[i] for i in perm]
    ids = torch.tensor(out[:n_blocks], dtype=torch.long)
    labels = labels[:n_blocks]

    comp = collections.Counter(labels)
    stats = {"docs_scanned": n_docs, "n_subsets": len(sets),
             "composition": dict(comp),
             "short_of_quota": {s: v for s, v in short.items() if v}}
    if verbose:
        print(f"[pile] scanned {n_docs} docs -> {ids.shape[0]} blocks of {seq} "
              f"from {len(sets)} subsets", flush=True)
        for s, v in comp.most_common():
            print(f"[pile]   {s:<28} {v:4d}  {100 * v / len(labels):5.1f}%",
                  flush=True)
        if stats["short_of_quota"]:
            print(f"[pile]   short of quota: {stats['short_of_quota']}",
                  flush=True)
    torch.save({"ids": ids, "labels": labels, "stats": stats}, f)
    return ids, labels, stats
