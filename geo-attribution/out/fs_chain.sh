#!/bin/bash
set -e
cd /workspace/param-clustering/geo-attribution
python3.12 stream67.py fitstream --device cuda:0 --C 4096 --sensor ig5 \
  --tag _fs --fs_tokens 80000000 --fs_stats_batches 60 --fs_cov_batches 240 \
  --pos_per_seq 64 --batch_blocks 32
echo "=== FITSTREAM DONE ==="
python3.12 stream67.py stream --device cuda:0 --C 4096 --sensor ig5 \
  --fit_tag _fs --tag _fs --quota 4 --batch_blocks 32 --pos_per_seq 4 \
  --shard 0 --nshard 2 --target_tokens 100000000 --log_every 5000 &
A=$!
python3.12 stream67.py stream --device cuda:1 --C 4096 --sensor ig5 \
  --fit_tag _fs --tag _fs --quota 4 --batch_blocks 32 --pos_per_seq 4 \
  --shard 1 --nshard 2 --target_tokens 100000000 --log_every 5000 &
B=$!
wait $A $B
echo "=== STREAM DONE ==="
python3.12 stream67.py bank --device cuda:0 --C 4096 --sensor ig5 \
  --tag _fs --n_samples 48 --refine_tokens 256 --out out/fitstream_C4096.json
echo "=== FS CHAIN DONE ==="
