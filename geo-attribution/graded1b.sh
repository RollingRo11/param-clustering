#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.12
D=/dev/shm/geo1b/run1
E="--tag run1 --eval_batches 4 --batch_seqs 4 --seq_len 512"
while pgrep -f "geo1b.py eval" > /dev/null; do sleep 20; done

echo "=== graded gating arms: a_c plot + PR (GPU1), ramp gamma 1 & 0.5 (GPU0) ==="
cp $D/banks_prop1b.pt $D/banks_prop1bb.pt
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY plot_ac1b.py &&
  $PY geo1b.py eval $E --banks_tag prop1b --gate_mode ramp --gate_thresh 0.02 --gate_tau 1.0 &&
  $PY geo1b.py eval $E --banks_tag prop1b --gate_mode ramp --gate_thresh 0.02 --gate_tau 0.5" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo1b.py eval $E --banks_tag prop1bb --gate_mode pr" &
wait
rm -f $D/banks_prop1bb.pt
mkdir -p out/full1b
cp $D/*.json out/full1b/ 2>/dev/null || true
echo "=== graded1b complete ==="
