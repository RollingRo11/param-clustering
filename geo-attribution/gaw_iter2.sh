#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12
E="--tag full65 --eval_batches 6 --batch_seqs 8 --seq_len 512"
D=out/full65

echo "=== arm A (GPU0): self-consistent iteration 2 | arm B (GPU1): s=16 support ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py extract_pg --tag full65 --src_banks_tag gawC1024 \
    --mass_banks_tag softC1024T1 --banks_tag gaw2C1024 \
    --gate_thresh 0.005 --gate_batches 12 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py eval $E --gate_thresh 0.02  --banks_tag gaw2C1024 &&
  mv $D/eval_gaw2C1024.json $D/eval_gaw2C1024_t02.json &&
  $PY geo67.py eval $E --gate_thresh 0.005 --banks_tag gaw2C1024 &&
  mv $D/eval_gaw2C1024.json $D/eval_gaw2C1024_t005.json &&
  $PY geo67.py eval $E --gate_thresh 0.001 --banks_tag gaw2C1024 &&
  mv $D/eval_gaw2C1024.json $D/eval_gaw2C1024_t001.json" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo67.py extract_ps --tag full65 --C 1024 --soft_T 1.0 --soft_s 16 \
    --banks_tag softC1024T1s16 &&
  $PY geo67.py extract_pg --tag full65 --src_banks_tag softC1024T1s16 \
    --banks_tag gawS16C1024 --gate_thresh 0.02 --gate_batches 12 \
    --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py eval $E --gate_thresh 0.02  --banks_tag gawS16C1024 &&
  mv $D/eval_gawS16C1024.json $D/eval_gawS16C1024_t02.json &&
  $PY geo67.py eval $E --gate_thresh 0.005 --banks_tag gawS16C1024 &&
  mv $D/eval_gawS16C1024.json $D/eval_gawS16C1024_t005.json" &
wait
for f in banks_gaw2C1024.pt:800000000 banks_softC1024T1s16.pt:1600000000 \
         banks_gawS16C1024.pt:1600000000; do
  n=${f%%:*}; min=${f##*:}; sz=$(stat -c%s $D/$n)
  if [ "$sz" -lt "$min" ]; then echo "QUOTA CHECK FAILED: $n only $sz"; exit 1; fi
  echo "$n OK ($sz bytes)"
done
echo "=== gaw_iter2 complete ==="
