#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.12
D=/dev/shm/geo1b/run1

echo "=== phase A: factor C=2048 (recomputes eigenpairs at k~2300) ==="
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py factor --tag run1 --C 2048

echo "=== phase B: proportional extraction at C=2048 ==="
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py extract_ps --tag run1 --C 2048 \
  --soft_T 1.0 --soft_s 8 --banks_tag propC2048
sz=$(stat -c%s $D/banks_propC2048.pt)
if [ "$sz" -lt 25000000000 ]; then echo "BANKS CHECK FAILED: $sz"; exit 1; fi
echo "banks_propC2048.pt OK ($sz bytes)"

echo "=== phase C: referees (GPU0) | hard eval with top-j (GPU1) ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY german1b.py --tag run1 --banks_tag propC2048 --gate_thresh 0.02 \
    --alphas 0.0 -1.0 --space weight &&
  $PY emote1b.py propC2048" &
CUDA_VISIBLE_DEVICES=1 $PY geo1b.py eval --tag run1 --eval_batches 2 \
  --batch_seqs 4 --seq_len 512 --gate_thresh 0.02 --banks_tag propC2048 &
wait
mkdir -p out/full1b
cp $D/*.json out/full1b/ 2>/dev/null || true
echo "=== c2048_1b complete ==="
