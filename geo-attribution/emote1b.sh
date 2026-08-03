#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== emote1b: emoticon component find + amplify + redirect (GPU0) ==="
CUDA_VISIBLE_DEVICES=0 /usr/bin/python3.12 emote1b.py
mkdir -p out/full1b
cp /dev/shm/geo1b/run1/emote1b.json out/full1b/
echo "=== emote1b complete ==="
