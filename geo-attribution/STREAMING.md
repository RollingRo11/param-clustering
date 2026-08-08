# Billion-position streaming decomposition

The large-scale path does not materialize a feature matrix or label vector for
the complete corpus. It has three bounded phases:

1. `collect` writes a fixed-size pilot feature matrix.
2. `fit_stream` learns and freezes its center, PCA projection, and spherical
   centroids.
3. `stream` assigns every later attribution vector online and retains only a
   stratified Algorithm-R reservoir of detailed `(pre, gradient)` tensors.

`overnight1b_stream.sh` runs the complete pilot, fit, billion-position stream,
and lazy extraction workflow. Its defaults are 131,072 pilot positions,
1,000,000,000 streamed positions, 2,048 clusters, and 16 detailed examples per
cluster globally:

```bash
bash overnight1b_stream.sh
```

The equivalent stages are:

```bash
CUDA_VISIBLE_DEVICES=0 python collect_fast.py spec \
  --tag stream_spec --feat_dim 16384

torchrun --nproc_per_node=2 collect_fast.py collect --profile optimized \
  --tag run1b_pilot --spec_tag stream_spec --n_positions 131072 \
  --pos_per_seq 506 --sub_per_seq 0 --batch_seqs 4 --seq_len 512 \
  --data_order sequential

CUDA_VISIBLE_DEVICES=0 python collect_fast.py fit_stream \
  --tag run1b --pilot_tag run1b_pilot --C 2048 --embed_dim 256 \
  --pilot_max_positions 131072

torchrun --nproc_per_node=2 collect_fast.py stream --profile optimized \
  --tag run1b --spec_tag stream_spec --stream_model_tag run1b \
  --n_positions 1000000000 --pos_per_seq 506 --batch_seqs 4 \
  --seq_len 512 --data_order sequential --C 2048 \
  --reservoir_per_cluster 16 --checkpoint_batches 128 --resume

CUDA_VISIBLE_DEVICES=0 python geo1b.py extract_ps --tag run1b --C 2048 \
  --soft_T 1.0 --soft_s 8 --banks_tag prop1b
```

`n_positions` counts attributed token positions, not input sequence tokens.
Increasing `pos_per_seq` batches more positions into each forward/backward pass;
it may be at most `seq_len - 6`. Sequential input order consumes disjoint rows
across ranks. The driver checks that the token file is long enough to avoid
wrapping. With 506 positions per 512-token sequence, 1B attributed positions
require about 1.012B input tokens. The file can be produced without accumulating
it in RAM:

```bash
python prep1b.py --target_tokens 1011859456
```

For Llama-3.2-1B, the default pilot feature files occupy 4 GiB total. The raw
BF16 extraction reservoirs occupy about 43 GiB total at `C=2048` and
`reservoir_per_cluster=16`, regardless of whether the assignment stream has
1M, 1B, or more positions. Reducing the reservoir count scales that storage
linearly.

Each rank checkpoints its exact input iterator, position RNG, reservoir RNG,
cluster counts, and reservoir slots atomically. Re-running the same command with
`--resume` continues from the latest compatible per-rank checkpoint. Completed
pilot fingerprints, frozen models, and stream shards are fingerprinted and
reused; incompatible configurations fail instead of being mixed.
