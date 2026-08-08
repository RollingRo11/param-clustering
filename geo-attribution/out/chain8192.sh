#!/bin/bash
set -e
cd /workspace/param-clustering/geo-attribution
until [ -f /dev/shm/geo67_stream/pile_1b_uint16.bin ]; do sleep 30; done
echo "=== PREP DONE ==="; date
python3.12 stream67.py fit --device cuda:0 --C 8192 --sensor ig5 \
  --fit_positions 65536 --pos_per_seq 64 --batch_blocks 8
echo "=== FIT DONE ==="; date
python3.12 stream67.py stream --device cuda:0 --C 8192 --sensor ig5 \
  --quota 4 --batch_blocks 32 --pos_per_seq 4 --shard 0 --nshard 2 \
  --target_tokens 1000000000 --log_every 2000 &
P0=$!
python3.12 stream67.py stream --device cuda:1 --C 8192 --sensor ig5 \
  --quota 4 --batch_blocks 32 --pos_per_seq 4 --shard 1 --nshard 2 \
  --target_tokens 1000000000 --log_every 2000 &
P1=$!
wait $P0 $P1
echo "=== STREAM DONE ==="; date
python3.12 stream67.py bank --device cuda:0 --C 8192 --sensor ig5 \
  --n_samples 48 --bank_chunk 32 --out out/stream67_C8192_ig5_1B.json
echo "=== BANK DONE ==="; date
