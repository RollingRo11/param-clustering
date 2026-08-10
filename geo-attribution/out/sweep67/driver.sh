#!/bin/bash
set -e
cd /workspace/param-clustering/geo-attribution
# wait for the archive move to free /dev/shm
until grep -q MOVED-OK out/archive_move.log 2>/dev/null; do sleep 30; done
echo "=== SHM FREED ==="; df -h /dev/shm | tail -1; date
# streams: both GPUs shard each count
for spec in "t100k 100000" "t10m 10000000" "t100m 100000000" "t500m 500000000"; do
  set -- $spec; tag=$1; n=$2
  python3.12 stream67.py stream --device cuda:0 --C 8192 --sensor ig5 \
    --quota 4 --batch_blocks 32 --pos_per_seq 4 --shard 0 --nshard 2 \
    --tag $tag --target_tokens $n --log_every 5000 &
  A=$!
  python3.12 stream67.py stream --device cuda:1 --C 8192 --sensor ig5 \
    --quota 4 --batch_blocks 32 --pos_per_seq 4 --shard 1 --nshard 2 \
    --tag $tag --target_tokens $n --log_every 5000 &
  B=$!
  wait $A $B
  echo "=== STREAM $tag DONE ==="; date
done
# banks: two at a time, one per GPU (tag "" = the existing 1B reservoirs)
bank() {
  python3.12 stream67.py bank --device cuda:$1 --C 8192 --sensor ig5 \
    --tag "$2" --n_samples 48 --refine_tokens 256 \
    --out out/sweep67/bank_${3}.json > out/sweep67/bank_${3}.log 2>&1
  echo "=== BANK $3 DONE ==="; date
}
bank 0 "t100k" t100k & bank 1 "t10m" t10m & wait
bank 0 "t100m" t100m & bank 1 "t500m" t500m & wait
bank 0 "" t1000m
echo "=== SWEEP COMPLETE ==="; date
