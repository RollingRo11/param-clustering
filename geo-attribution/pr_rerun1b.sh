#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "german_k4_1b.py" > /dev/null || pgrep -f "gate_mode pr" > /dev/null; do sleep 20; done
echo "=== pr_rerun1b: overflow-fixed PR gating eval (GPU1) ==="
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3.12 geo1b.py eval --tag run1 \
  --eval_batches 4 --batch_seqs 4 --seq_len 512 --banks_tag prop1b --gate_mode pr
mkdir -p out/full1b
cp /dev/shm/geo1b/run1/eval_prop1b_pr.json out/full1b/
echo "=== pr_rerun1b complete ==="
