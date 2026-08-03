#!/bin/bash
# GIM sensor A/B arm: identical config to the N=32k baseline (tag full, C=512,
# r8, thresh 0.02) except --sensor gim in collect. Judged on the induction
# canary + full battery vs out/full/*_part8.
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12
TAG=gim32
$PY -m torch.distributed.run --standalone --nproc_per_node=2 geo67.py collect \
  --tag $TAG --sensor gim --n_positions 32768 --pos_per_seq 16 --batch_seqs 16 --seq_len 512
sleep 30
$PY -m torch.distributed.run --standalone --nproc_per_node=2 geo67.py gram --tag $TAG --gram_block 256
CUDA_VISIBLE_DEVICES=0 $PY geo67.py factor --tag $TAG --C 512
CUDA_VISIBLE_DEVICES=0 $PY geo67.py extract_p --tag $TAG --C 512 --base_ratio 8 --banks_tag gim_r8 &
CUDA_VISIBLE_DEVICES=1 $PY geo67.py extract_p --tag $TAG --C 512 --base_ratio 5 --banks_tag gim_r5 &
wait
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py eval --tag $TAG --banks_tag gim_r8 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py canary --tag $TAG --banks_tag gim_r8 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256 &&
  $PY german67.py --tag $TAG --banks_tag gim_r8 --gate_thresh 0.02 --alphas 0.0 -1.0 -2.0" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo67.py eval --tag $TAG --banks_tag gim_r5 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py canary --tag $TAG --banks_tag gim_r5 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256" &
wait
echo "=== gim arm complete ==="
