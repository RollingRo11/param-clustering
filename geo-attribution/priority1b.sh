#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.12
D=/dev/shm/geo1b/run1
E="--tag run1 --eval_batches 4 --batch_seqs 4 --seq_len 512"

echo "=== GPU1: german weight edit -> PR gating | GPU0: a_c plot -> ramp gating ==="
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY german1b.py --tag run1 --banks_tag prop1b --gate_thresh 0.02 \
    --alphas 0.0 -1.0 --space weight &&
  $PY geo1b.py eval $E --banks_tag prop1b --gate_mode pr" &
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY plot_ac1b.py &&
  $PY geo1b.py eval $E --banks_tag prop1b --gate_mode ramp --gate_thresh 0.02 --gate_tau 1.0 &&
  $PY geo1b.py eval $E --banks_tag prop1b --gate_mode ramp --gate_thresh 0.02 --gate_tau 0.5" &
wait
mkdir -p out/full1b
cp $D/*.json out/full1b/ 2>/dev/null || true
echo "=== priority1b complete ==="
