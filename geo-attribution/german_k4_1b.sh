#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "gate_mode pr" > /dev/null; do sleep 20; done
echo "=== german_k4_1b: alpha sweep + solo edits + ownership profiles (GPU1) ==="
CUDA_VISIBLE_DEVICES=1 /usr/bin/python3.12 german_k4_1b.py
mkdir -p out/full1b
cp /dev/shm/geo1b/run1/german_k4_1b.json out/full1b/
echo "=== german_k4_1b complete ==="
