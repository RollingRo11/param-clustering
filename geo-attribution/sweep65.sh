#!/bin/bash
# Scale + sweep: N=65536 corpus, C in {512,1024,2048} x base_ratio in {8,5}.
# Phases: collect+gram (DDP both GPUs) -> factor (shared eig cache) ->
# extract_p arms -> per-arm referee battery (eval, canary, german gate+weight).
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12
TAG=full65

echo "=== phase A: collect + gram (DDP) ==="
$PY -m torch.distributed.run --standalone --nproc_per_node=2 geo67.py collect \
  --tag $TAG --n_positions 65536 --pos_per_seq 16 --batch_seqs 16 --seq_len 512
sleep 30
$PY -m torch.distributed.run --standalone --nproc_per_node=2 geo67.py gram \
  --tag $TAG --gram_block 256

echo "=== phase B: factor (eig once at kmax=2304, then per-C kmeans) ==="
CUDA_VISIBLE_DEVICES=0 $PY geo67.py factor --tag $TAG --C 2048
CUDA_VISIBLE_DEVICES=0 $PY geo67.py factor --tag $TAG --C 1024 &
CUDA_VISIBLE_DEVICES=1 $PY geo67.py factor --tag $TAG --C 512 &
wait

echo "=== phase C: partition extraction arms ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py extract_p --tag $TAG --C 512  --base_ratio 8 --banks_tag C512_r8 &&
  $PY geo67.py extract_p --tag $TAG --C 1024 --base_ratio 8 --banks_tag C1024_r8 &&
  $PY geo67.py extract_p --tag $TAG --C 2048 --base_ratio 8 --banks_tag C2048_r8" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo67.py extract_p --tag $TAG --C 512  --base_ratio 5 --banks_tag C512_r5 &&
  $PY geo67.py extract_p --tag $TAG --C 1024 --base_ratio 5 --banks_tag C1024_r5 &&
  $PY geo67.py extract_p --tag $TAG --C 2048 --base_ratio 5 --banks_tag C2048_r5" &
wait

echo "=== phase D: referee battery per arm ==="
battery () {  # $1 = banks_tag
  $PY geo67.py eval   --tag $TAG --banks_tag $1 --gate_thresh 0.02 \
      --eval_batches 6 --batch_seqs 8 --seq_len 512
  $PY geo67.py canary --tag $TAG --banks_tag $1 --gate_thresh 0.02 \
      --batch_seqs 16 --seq_len 256
  $PY german67.py --tag $TAG --banks_tag $1 --gate_thresh 0.02 --alphas 0.0 -1.0 -2.0
  $PY german67.py --tag $TAG --banks_tag $1 --gate_thresh 0.02 --alphas 0.0 -1.0 \
      --space weight
}
export -f battery; export PY TAG
CUDA_VISIBLE_DEVICES=0 bash -c "battery C512_r8 && battery C1024_r8 && battery C2048_r8" &
CUDA_VISIBLE_DEVICES=1 bash -c "battery C512_r5 && battery C1024_r5 && battery C2048_r5" &
wait
echo "=== sweep complete ==="
