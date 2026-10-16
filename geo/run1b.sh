#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
PY=/usr/bin/python3.12
D=/dev/shm/geo1b/run1
E="--tag run1 --eval_batches 4 --batch_seqs 4 --seq_len 512"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f $D/collect_rank1.pt ] && [ $(stat -c%s $D/collect_rank1.pt) -ge 20000000000 ]; then
  echo "=== phase A: collect shards present, skipping ==="
else
  echo "=== phase A: collect (DDP, N=16384) ==="
  torchrun --nproc_per_node=2 geo1b.py collect --tag run1 --n_positions 16384 \
    --pos_per_seq 16 --batch_seqs 8 --seq_len 512
  for r in 0 1; do
    sz=$(stat -c%s $D/collect_rank$r.pt)
    if [ "$sz" -lt 20000000000 ]; then
      echo "COLLECT SHARD CHECK FAILED: rank$r only $sz bytes"; exit 1; fi
    echo "collect_rank$r.pt OK ($sz bytes)"
  done
fi

if [ -f $D/gram.pt ] && [ $(stat -c%s $D/gram.pt) -ge 1000000000 ]; then
  echo "=== phase B: gram.pt present, skipping ==="
else
  echo "=== phase B: gram (DDP, module-sharded) ==="
  torchrun --nproc_per_node=2 geo1b.py gram --tag run1 --gram_block 256
  sz=$(stat -c%s $D/gram.pt)
  if [ "$sz" -lt 1000000000 ]; then echo "GRAM CHECK FAILED: $sz"; exit 1; fi
  echo "gram.pt OK ($sz bytes)"
fi

if [ -f $D/labels_C512.pt ]; then
  echo "=== phase C: labels present, skipping ==="
else
  echo "=== phase C: factor C=512 ==="
  CUDA_VISIBLE_DEVICES=0 $PY geo1b.py factor --tag run1 --C 512
fi

if [ -f $D/banks_prop1b.pt ] && [ $(stat -c%s $D/banks_prop1b.pt) -ge 25000000000 ]; then
  echo "=== phase D: banks present, skipping ==="
else
  echo "=== phase D: proportional softpart extraction ==="
  CUDA_VISIBLE_DEVICES=0 $PY geo1b.py extract_ps --tag run1 --C 512 \
    --soft_T 1.0 --soft_s 8 --banks_tag prop1b
  sz=$(stat -c%s $D/banks_prop1b.pt)
  if [ "$sz" -lt 25000000000 ]; then echo "BANKS CHECK FAILED: $sz"; exit 1; fi
  echo "banks_prop1b.pt OK ($sz bytes)"
fi

echo "=== phase E: proportional evals at two densities ==="
cp $D/banks_prop1b.pt $D/banks_prop1bb.pt
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo1b.py eval $E --gate_thresh 0.02 --banks_tag prop1b &&
  mv $D/eval_prop1b.json $D/eval_prop1b_t02.json" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo1b.py eval $E --gate_thresh 0.005 --banks_tag prop1bb &&
  mv $D/eval_prop1bb.json $D/eval_prop1b_t005.json" &
wait
rm -f $D/banks_prop1bb.pt

mkdir -p out/full1b
cp $D/*.json out/full1b/ 2>/dev/null || true
cp $D/spectrum.pt out/full1b/ 2>/dev/null || true
echo "=== run1b complete (jsons copied to out/full1b/) ==="
