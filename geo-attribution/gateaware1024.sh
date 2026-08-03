#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12
E="--tag full65 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512"
D=out/full65

echo "=== phase A: proportional softpart at C=1024 (bootstrap + baseline) ==="
CUDA_VISIBLE_DEVICES=0 $PY geo67.py extract_ps --tag full65 --C 1024 \
  --soft_T 1.0 --soft_s 8 --banks_tag softC1024T1
sz=$(stat -c%s $D/banks_softC1024T1.pt)
if [ "$sz" -lt 800000000 ]; then
  echo "QUOTA CHECK FAILED: banks_softC1024T1.pt only $sz bytes"; exit 1
fi
echo "banks_softC1024T1.pt OK ($sz bytes)"

echo "=== phase B: gate-aware solve (GPU0) | baseline battery (GPU1) ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py extract_pg --tag full65 --src_banks_tag softC1024T1 \
    --banks_tag gawC1024 --gate_thresh 0.02 --gate_batches 12 \
    --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py eval $E --banks_tag gawC1024" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo67.py eval $E --banks_tag softC1024T1 &&
  $PY geo67.py canary --tag full65 --banks_tag softC1024T1 --gate_thresh 0.02 \
    --batch_seqs 16 --seq_len 256 &&
  $PY german67.py --tag full65 --banks_tag softC1024T1 --gate_thresh 0.02 \
    --alphas 0.0 -1.0 --space weight" &
wait
sz=$(stat -c%s $D/banks_gawC1024.pt)
if [ "$sz" -lt 800000000 ]; then
  echo "QUOTA CHECK FAILED: banks_gawC1024.pt only $sz bytes"; exit 1
fi
echo "banks_gawC1024.pt OK ($sz bytes)"

echo "=== phase C: gate-aware battery ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py canary --tag full65 --banks_tag gawC1024 --gate_thresh 0.02 \
    --batch_seqs 16 --seq_len 256 &&
  $PY german67.py --tag full65 --banks_tag gawC1024 --gate_thresh 0.02 \
    --alphas 0.0 -1.0 --space weight" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY german67.py --tag full65 --banks_tag gawC1024 --gate_thresh 0.02 \
    --alphas 0.0 -1.0" &
wait
echo "=== gateaware1024 complete ==="
