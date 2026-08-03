#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12
E="--tag full65 --eval_batches 6 --batch_seqs 8 --seq_len 512"
D=out/full65
while pgrep -f gaw_iter2.sh > /dev/null; do sleep 10; done

echo "=== combo: iteration 2 on s=16 supports (GPU0) | german probes (GPU1) ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py extract_pg --tag full65 --src_banks_tag gawS16C1024 \
    --mass_banks_tag softC1024T1s16 --banks_tag gaw2S16C1024 \
    --gate_thresh 0.005 --gate_batches 12 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py eval $E --gate_thresh 0.005 --banks_tag gaw2S16C1024 &&
  mv $D/eval_gaw2S16C1024.json $D/eval_gaw2S16C1024_t005.json &&
  $PY geo67.py eval $E --gate_thresh 0.001 --banks_tag gaw2S16C1024 &&
  mv $D/eval_gaw2S16C1024.json $D/eval_gaw2S16C1024_t001.json &&
  $PY geo67.py eval $E --gate_thresh 0.02 --banks_tag gaw2S16C1024 &&
  mv $D/eval_gaw2S16C1024.json $D/eval_gaw2S16C1024_t02.json" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY german67.py --tag full65 --banks_tag gaw2C1024 --gate_thresh 0.02 \
    --alphas 0.0 -1.0 --space weight" &
wait
sz=$(stat -c%s $D/banks_gaw2S16C1024.pt)
if [ "$sz" -lt 1600000000 ]; then echo "QUOTA CHECK FAILED: $sz"; exit 1; fi
echo "banks_gaw2S16C1024.pt OK ($sz bytes)"
echo "=== gaw_combo complete ==="
