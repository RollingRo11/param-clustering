#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12
arm () {  # $1 = ratio
  $PY geo67.py extract_p --tag full65 --C 512 --base_ratio $1 --banks_tag C512_r$1
  $PY geo67.py eval   --tag full65 --banks_tag C512_r$1 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512
  $PY geo67.py canary --tag full65 --banks_tag C512_r$1 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256
  $PY german67.py --tag full65 --banks_tag C512_r$1 --gate_thresh 0.02 --alphas 0.0 -1.0
  $PY german67.py --tag full65 --banks_tag C512_r$1 --gate_thresh 0.02 --alphas 0.0 -1.0 --space weight
}
export -f arm; export PY
CUDA_VISIBLE_DEVICES=0 bash -c "arm 4 && arm 2" &
CUDA_VISIBLE_DEVICES=1 bash -c "arm 3" &
wait
echo "=== deep-r arms complete ==="
