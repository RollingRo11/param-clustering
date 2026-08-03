#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.12
# wait for the PR-gating eval (GPU1, banks_tag prop1bb) to release its memory
while pgrep -f "banks_tag prop1bb" > /dev/null; do sleep 20; done

echo "=== german1b: weight-space German erasure on prop1b (GPU1) ==="
CUDA_VISIBLE_DEVICES=1 $PY german1b.py --tag run1 --banks_tag prop1b \
  --gate_thresh 0.02 --alphas 0.0 -1.0 --space weight
mkdir -p out/full1b
cp /dev/shm/geo1b/run1/german_*.json out/full1b/ 2>/dev/null || true
echo "=== german1b complete ==="
