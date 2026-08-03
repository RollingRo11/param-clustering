#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12
E="--tag full65 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512"

echo "=== exp1: soft gates on existing hard-partition banks ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  for tau in 0.5 1.0 2.0; do
    $PY geo67.py eval $E --banks_tag C512_r2 --gate_mode soft --gate_tau \$tau
    $PY geo67.py eval $E --banks_tag C512_r8 --gate_mode soft --gate_tau \$tau
  done" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  for tau in 0.5 1.0 2.0; do
    $PY geo67.py eval $E --banks_tag C512_r3 --gate_mode soft --gate_tau \$tau
  done
  echo '=== exp2 extract: fractional ownership T=1 and T=0.5 ==='
  $PY geo67.py extract_ps --tag full65 --C 512 --soft_T 1.0 --soft_s 8 --banks_tag softT1" &
wait
CUDA_VISIBLE_DEVICES=0 $PY geo67.py extract_ps --tag full65 --C 512 --soft_T 0.5 --soft_s 8 --banks_tag softT05 &
wait

echo "=== exp2 batteries ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py eval $E --banks_tag softT1 &&
  $PY geo67.py eval $E --banks_tag softT1 --gate_mode soft --gate_tau 1.0 &&
  $PY geo67.py eval $E --banks_tag softT1 --gate_mode soft --gate_tau 2.0 &&
  $PY geo67.py canary --tag full65 --banks_tag softT1 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256 &&
  $PY german67.py --tag full65 --banks_tag softT1 --gate_thresh 0.02 --alphas 0.0 -1.0 &&
  $PY german67.py --tag full65 --banks_tag softT1 --gate_thresh 0.02 --alphas 0.0 -1.0 --space weight" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo67.py eval $E --banks_tag softT05 &&
  $PY geo67.py eval $E --banks_tag softT05 --gate_mode soft --gate_tau 1.0 &&
  $PY geo67.py eval $E --banks_tag softT05 --gate_mode soft --gate_tau 2.0 &&
  $PY german67.py --tag full65 --banks_tag softT05 --gate_thresh 0.02 --alphas 0.0 -1.0 --space weight" &
wait
echo "=== soft experiments complete ==="
