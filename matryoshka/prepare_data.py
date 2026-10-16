"""Shard-stratified, without-replacement sample of VPD's globally shuffled Pile.

Select a uniformly located cyclic window within each shard. Because rows were
globally shuffled by the dataset authors, a window does not follow document
order. Every shard contributes; row inclusion probabilities within a shard are
equal. Store row origins and shuffle the combined sample before either trainer
reads it. No claim of source-category balancing: those labels are not published.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import fsspec
import aiohttp
from huggingface_hub import HfApi
import numpy as np
import pyarrow.parquet as pq

from common import ROOT, DATASET, POSITIONS, CONTEXT, credentials, prepare_target, save_json, sha, upload


def sample_shard(path, revision, quota, seed, output):
    output = Path(output)
    meta_path = output.with_suffix('.json')
    if output.exists() and meta_path.exists():
        info = json.loads(meta_path.read_text())
        assert info['quota'] == quota and info['revision'] == revision
        return info
    rng = np.random.default_rng(seed)
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{path}'
    last = None
    for attempt in range(6):
        try:
            headers = {'Authorization': 'Bearer '+os.environ['HF_TOKEN']} if os.environ.get('HF_TOKEN') else {}
            with fsspec.open(url, 'rb', block_size=64<<10, cache_type='readahead', headers=headers,
                             client_kwargs={'timeout': aiohttp.ClientTimeout(total=60)}) as f:
                parquet = pq.ParquetFile(f)
                n = parquet.metadata.num_rows
                assert quota <= n
                start = int(rng.integers(n))
                indices = (np.arange(quota)+start)%n
                sizes = np.array([parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)])
                ends = sizes.cumsum(); begins = ends-sizes
                groups = np.searchsorted(ends, indices, side='right')
                result = np.empty((quota, 513), dtype=np.uint16)
                for group in np.unique(groups):
                    table = parquet.read_row_group(int(group), columns=['input_ids'])
                    column = table['input_ids'].combine_chunks()
                    assert np.all(np.diff(column.offsets.to_numpy()) == 513)
                    rows = column.values.to_numpy().reshape(-1, 513)
                    select = groups == group
                    selected = rows[indices[select]-begins[group]]
                    assert selected.min() >= 0 and selected.max() < 50277
                    result[select] = selected
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix('.tmp')
            result.tofile(temp); os.replace(temp, output)
            info = dict(path=path, revision=revision, rows=n, start=start, quota=quota,
                        row_groups=list(map(int, np.unique(groups))), sha256=sha(output))
            save_json(info, meta_path)
            return info
        except Exception as exc:
            last = exc
            print(json.dumps(dict(retry=attempt+1, path=path, error=str(exc)[:300])), flush=True)
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f'Failed sampling {path}: {last}') from last


def prepare_split(paths, revision, count, split, workers):
    stage = ROOT/'data/strata'/split
    stage.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260910 if split == 'train' else 20260911)
    quotas = np.full(len(paths), count//len(paths), dtype=np.int64)
    quotas[rng.permutation(len(paths))[:count%len(paths)]] += 1
    infos = [None]*len(paths)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(sample_shard, path, revision, int(q), 17001+i,
                               stage/f'{i:05d}.uint16'): i for i, (path, q) in enumerate(zip(paths, quotas))}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]; infos[i] = future.result()
            if done % 25 == 0 or done == len(paths):
                print(json.dumps(dict(split=split, shards_done=done, shards_total=len(paths),
                                      elapsed_seconds=time.monotonic()-started)), flush=True)
    # HF shards are almost equal in row count. Assert that equal-shard quotas
    # remain within rounding error of proportional allocation; don't silently
    # overrepresent small shards if upstream changes the layout.
    n = np.array([info['rows'] for info in infos])
    ideal = count*n/n.sum()
    if np.max(np.abs(quotas-ideal)) > 2:
        raise ValueError('Unequal upstream shard sizes require proportional resampling')
    concat = ROOT/'data'/f'{split}.unshuffled.uint16'
    with open(concat, 'wb') as out:
        for i in range(len(paths)):
            with open(stage/f'{i:05d}.uint16', 'rb') as f:
                while block := f.read(8<<20): out.write(block)
    data = np.memmap(concat, mode='r', dtype=np.uint16, shape=(count, 513))
    order = rng.permutation(count)
    final = ROOT/'data'/f'{split}.uint16'
    with open(final.with_suffix('.tmp'), 'wb') as out:
        for i in range(0, count, 16384): data[order[i:i+16384]].tofile(out)
    os.replace(final.with_suffix('.tmp'), final)
    np.save(ROOT/'data'/f'{split}.permutation.npy', order)
    del data
    concat.unlink()
    save_json(infos, ROOT/'data'/f'{split}.strata.json')
    return dict(file=final.name, rows=count, positions=count*CONTEXT, sha256=sha(final),
                shards=len(paths), shard_rows_min=int(n.min()), shard_rows_max=int(n.max()),
                quota_min=int(quotas.min()), quota_max=int(quotas.max()),
                strata_sha256=sha(ROOT/'data'/f'{split}.strata.json'),
                permutation_sha256=sha(ROOT/'data'/f'{split}.permutation.npy'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    credentials(); prepare_target()
    api = HfApi()
    revision = api.dataset_info(DATASET).sha
    files = api.list_repo_files(DATASET, repo_type='dataset', revision=revision)
    paths = {s: sorted(p for p in files if p.startswith('data/'+s+'-') and p.endswith('.parquet'))
             for s in ('train', 'val')}
    assert len(paths['train']) == 2021
    manifest = dict(dataset=DATASET, revision=revision, context=512, row_tokens=513,
                    sampling='equal/proportional shard quotas; uniform cyclic windows in globally shuffled rows; shared final permutation',
                    source_category_labels_available=False, unique_training_rows=True)
    manifest['val'] = prepare_split(paths['val'], revision, 2048, 'val', args.workers)
    save_json(manifest, ROOT/'data/manifest.json')
    manifest['train'] = prepare_split(paths['train'], revision, POSITIONS//CONTEXT, 'train', args.workers)
    save_json(manifest, ROOT/'data/manifest.json')
    uploaded = []
    for filename in ('manifest.json', 'train.strata.json', 'val.strata.json', 'train.permutation.npy',
                     'val.permutation.npy', 'train.uint16', 'val.uint16'):
        uploaded.append(upload(ROOT/'data'/filename, 'data/'+filename))
    save_json(uploaded, ROOT/'output/data_r2.json')
    print('DATA_READY', flush=True)


if __name__ == '__main__': main()
