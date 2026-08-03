#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
PY=/usr/bin/python3.12

echo "=== smoke1b: miniature full chain on Llama-3.2-1B (N=512, C=32) ==="
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py collect --tag smoke --n_positions 512 \
  --pos_per_seq 16 --batch_seqs 8 --seq_len 512
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py gram --tag smoke --gram_block 256
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py factor --tag smoke --C 32
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py extract_ps --tag smoke --C 32 \
  --soft_T 1.0 --soft_s 8 --banks_tag sm
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py eval --tag smoke --banks_tag sm \
  --gate_thresh 0.02 --eval_batches 1 --batch_seqs 4 --seq_len 512
echo "=== smoke1b complete ==="
