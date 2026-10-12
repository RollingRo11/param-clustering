"""Reproduce, row for row, the training data of VPD's published run goodfire/spd/s-55ea3f9b.

Pinned from that run (par-auto notes, 2026-09-12):
  code      goodfire-ai/spd dev@a8919d47 (2026-02-25 15:06 UTC, 14 min before the run was created;
            its message names the run's pretrained model t-9d2b8f02). spd/data.py there is byte-identical
            to main@4c0a470e.
  data      danbraunai/pile-uncopyrighted-tok-shuffled @ 34bf30a7, the last data upload before the run.
            Later commits add only a README, but a dataset card can change file resolution, so pin this.
  libs      datasets 4.4.2, huggingface-hub 0.36.0, fsspec 2025.10.0, pyarrow 22.0.0 (run's requirements.txt)
  launch    8 DDP ranks (all 8 node GPUs busy in the run's W&B system metrics); batch_size 64 -> 8 per rank.

Per rank, spd.data.create_data_loader does:
  load_dataset(streaming=True, split='train') -> .shard(num_shards=8, index=rank)    [2021 files >= 8]
  -> .shuffle(seed=0, buffer_size=1000) -> .map(truncate to 512) -> torch format
  -> DataLoader(batch_size=8, drop_last=True)
and run_spd.py draws one sample batch, then one batch per step for steps+1 = 400,001 steps, skipping the
optimizer update on the last. The 400,000 updates used per-rank batches 1..400,000.

Writes, per rank, those batches in order as full 513-token rows (3,200,000 x 513, uint16). The truncating
map keeps row order; the kept final token is the true next token that our loss needs.
Check: batches 0-2 are rebuilt through VPD's exact truncating DataLoader; batch 0 must match the skipped
rows and batches 1-2 must equal our first 16 rows' first 512 tokens.

  python prepare_vpd_stream.py --out DIR --rank R        (one process per rank, run all 8 in parallel)
  python prepare_vpd_stream.py --out DIR --finalize      (verifies all ranks, writes manifest.json)
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

NAME = 'danbraunai/pile-uncopyrighted-tok-shuffled'
REVISION_PREFIX = '34bf30a7a4f8'
WORLD, RANK_BATCH, SEED, BUFFER, N_CTX, ROW = 8, 8, 0, 1000, 512, 513
BATCHES = int(os.environ.get('PREP_BATCHES', '400000'))                # override only for a local test
PINNED = {'datasets': '4.4.2', 'huggingface_hub': '0.36.0', 'fsspec': '2025.10.0', 'pyarrow': '22.0.0'}


def check_versions():
    import datasets, huggingface_hub, fsspec, pyarrow
    found = dict(datasets=datasets.__version__, huggingface_hub=huggingface_hub.__version__,
                 fsspec=fsspec.__version__, pyarrow=pyarrow.__version__)
    assert found == PINNED, f'library versions differ from the published run: {found}'
    return found


def revision():
    from huggingface_hub import HfApi
    full = [c.commit_id for c in HfApi().list_repo_commits(NAME, repo_type='dataset') if c.commit_id.startswith(REVISION_PREFIX)]
    assert len(full) == 1, full
    return full[0]


def pipeline(rank, rev):
    """VPD's shard -> shuffle order for one rank, before its truncating map."""
    import datasets
    from datasets import load_dataset
    for key, value in (('STREAMING_READ_MAX_RETRIES', 50), ('STREAMING_READ_RETRY_INTERVAL', 10)):
        if hasattr(datasets.config, key): setattr(datasets.config, key, value)
    ds = load_dataset(NAME, streaming=True, split='train', revision=rev, trust_remote_code=False)
    assert isinstance(ds.num_shards, int) and ds.num_shards >= WORLD, ds.num_shards
    return ds.shard(num_shards=WORLD, index=rank).shuffle(seed=SEED, buffer_size=BUFFER)


def exact_batches(rank, rev, n):
    """VPD's own DataLoader for this rank: truncating map, torch format, batch 8, drop_last."""
    import torch
    from torch.utils.data import DataLoader
    torch_ds = pipeline(rank, rev).map(lambda x: {'input_ids': x['input_ids'][:N_CTX]}).with_format('torch')
    generator = torch.Generator(device='cpu'); generator.manual_seed(SEED)
    batches = iter(DataLoader(torch_ds, batch_size=RANK_BATCH, shuffle=False, drop_last=True, generator=generator))
    return [next(batches)['input_ids'].numpy().astype(np.int64) for _ in range(n)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(64 << 20), b''): h.update(block)
    return h.hexdigest()


def write_rank(rank, out):
    versions = check_versions(); rev = revision(); started = time.time()
    path, total = out/f'rank{rank}.uint16', BATCHES*RANK_BATCH
    tmp = path.with_suffix('.partial')
    rows = np.memmap(tmp, mode='w+', dtype=np.uint16, shape=(total, ROW))
    skipped, filled = [], 0
    for chunk in pipeline(rank, rev).iter(batch_size=4096):
        arr = np.asarray(chunk['input_ids'], dtype=np.int64)
        assert arr.ndim == 2 and arr.shape[1] == ROW and arr.min() >= 0 and arr.max() < 65536, arr.shape
        if sum(len(s) for s in skipped) < RANK_BATCH:            # batch 0 went to run_spd.py's sample_batch
            need = RANK_BATCH-sum(len(s) for s in skipped)
            skipped.append(arr[:need]); arr = arr[need:]
        n = min(len(arr), total-filled)
        rows[filled:filled+n] = arr[:n].astype(np.uint16); filled += n
        if filled // 400_000 != (filled-n) // 400_000:
            print(json.dumps(dict(rank=rank, rows=filled, seconds=round(time.time()-started))), flush=True)
        if filled == total: break
    assert filled == total, (rank, filled)
    rows.flush(); del rows
    os.replace(tmp, path)
    exact = exact_batches(rank, rev, 3)
    mine = np.memmap(path, mode='r', dtype=np.uint16, shape=(total, ROW))
    match = (np.array_equal(exact[0], np.concatenate(skipped)[:, :N_CTX])
             and np.array_equal(exact[1], mine[0:8, :N_CTX].astype(np.int64))
             and np.array_equal(exact[2], mine[8:16, :N_CTX].astype(np.int64)))
    record = dict(rank=rank, revision=rev, versions=versions, rows=total, row_tokens=ROW, first_batch_index=1,
                  exact_loader_match=bool(match), sha256=sha256(path), seconds=round(time.time()-started))
    (out/f'rank{rank}.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(record), flush=True)
    assert match, 'rows differ from VPD\'s exact DataLoader'


def finalize(out):
    ranks = [json.loads((out/f'rank{r}.json').read_text()) for r in range(WORLD)]
    assert all(r['exact_loader_match'] for r in ranks) and len({r['revision'] for r in ranks}) == 1
    for r in ranks: assert sha256(out/f'rank{r["rank"]}.uint16') == r['sha256'], r['rank']
    manifest = dict(source_run='goodfire/spd/s-55ea3f9b', code='goodfire-ai/spd dev@a8919d4744becc789e6a0dc8de3d50d1d01ace63',
                    dataset=NAME, revision=ranks[0]['revision'], world_size=WORLD, rows_per_rank_batch=RANK_BATCH,
                    batches_per_rank=BATCHES, first_batch_index=1, row_tokens=ROW, shard_then_shuffle=True,
                    seed=SEED, buffer_size=BUFFER, versions=ranks[0]['versions'], ranks=ranks, complete=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1))
    print('VPD_STREAM_MANIFEST_WRITTEN', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rank', type=int); ap.add_argument('--finalize', action='store_true')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    finalize(args.out) if args.finalize else write_rank(args.rank, args.out)


if __name__ == '__main__': main()
