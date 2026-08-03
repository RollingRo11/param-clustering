#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
export HF_HOME=/dev/shm/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.12
D=/dev/shm/geo1b/run1lp
while pgrep -f "c2048_1b.sh" > /dev/null; do sleep 30; done

avail=$(df --output=avail -B1 /dev/shm | tail -1)
if [ "$avail" -lt 90000000000 ]; then echo "SHM SPACE CHECK FAILED: $avail"; exit 1; fi

echo "=== lp1b phase A: collect with scalar=logp_pred (DDP, N=16384) ==="
torchrun --nproc_per_node=2 geo1b.py collect --tag run1lp --scalar logp_pred \
  --n_positions 16384 --pos_per_seq 16 --batch_seqs 8 --seq_len 512
for r in 0 1; do
  sz=$(stat -c%s $D/collect_rank$r.pt)
  if [ "$sz" -lt 20000000000 ]; then echo "COLLECT CHECK FAILED: rank$r $sz"; exit 1; fi
  echo "collect_rank$r.pt OK ($sz bytes)"
done

echo "=== lp1b phase B: gram ==="
torchrun --nproc_per_node=2 geo1b.py gram --tag run1lp --gram_block 256
echo "=== lp1b phase C: factor C=512 ==="
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py factor --tag run1lp --C 512
echo "=== lp1b phase D: extraction ==="
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py extract_ps --tag run1lp --C 512 \
  --soft_T 1.0 --soft_s 8 --banks_tag propLP
sz=$(stat -c%s $D/banks_propLP.pt)
if [ "$sz" -lt 25000000000 ]; then echo "BANKS CHECK FAILED: $sz"; exit 1; fi
echo "banks_propLP.pt OK ($sz bytes)"

echo "=== lp1b phase E: german referee (GPU0) | hard eval (GPU1) ==="
CUDA_VISIBLE_DEVICES=0 $PY german1b.py --tag run1lp --banks_tag propLP \
  --gate_thresh 0.02 --alphas 0.0 -1.0 --space weight &
CUDA_VISIBLE_DEVICES=1 $PY geo1b.py eval --tag run1lp --eval_batches 2 \
  --batch_seqs 4 --seq_len 512 --gate_thresh 0.02 --banks_tag propLP &
wait
mkdir -p out/full1b
cp $D/*.json out/full1b/ 2>/dev/null || true
echo "=== lp1b complete ==="
