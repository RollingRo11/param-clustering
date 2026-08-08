#!/bin/bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PYTHON:-python}
ARTIFACT_ROOT=${GEO_ATTRIBUTION_ARTIFACT_ROOT:-/dev/shm/geo1b}
D=$ARTIFACT_ROOT/run1m
R1=$ARTIFACT_ROOT/run1

while pgrep -f "autointerp1b.py" > /dev/null; do sleep 60; done

echo "=== phase 0: feature spec + validation against exact 16k gram ==="
if [ -f "$D/spec.pt" ] && [ -f "$D/validate.json" ]; then
  echo "validated feature spec present, reusing"
else
  CUDA_VISIBLE_DEVICES=1 $PY geo1m.py spec --tag run1m --feat_dim 16384
  CUDA_VISIBLE_DEVICES=1 $PY geo1m.py validate --tag run1m --val_n 2048 --val_gate 0.75
fi

echo "=== phase 0b: shm cleanup (regenerable N=16k intermediates) ==="
rm -f $R1/collect_rank0.pt $R1/collect_rank1.pt $R1/banks_prop1b.pt
avail=$(df --output=avail -B1 /dev/shm | tail -1)
echo "shm avail after cleanup: $avail"
if [ "$avail" -lt 160000000000 ]; then echo "SHM SPACE CHECK FAILED"; exit 1; fi

echo "=== phase A: BF16/Flash-GIM collect N=1,048,576 (reusable fingerprints) ==="
torchrun --nproc_per_node=2 collect_fast.py collect --profile optimized \
  --tag run1m --spec_tag run1m --n_positions 1048576 --pos_per_seq 64 \
  --sub_per_seq 2 --batch_seqs 8 --seq_len 512
for r in 0 1; do
  fs=$(stat -c%s $D/feat_rank$r.f16)
  if [ "$fs" -lt 17000000000 ]; then echo "FEAT CHECK FAILED: rank$r $fs"; exit 1; fi
  cs=$(stat -c%s $D/collect_rank$r.pt)
  if [ "$cs" -lt 20000000000 ]; then echo "SUB CHECK FAILED: rank$r $cs"; exit 1; fi
  echo "rank$r OK (feat $fs, sub $cs)"
done

echo "=== phase B: cluster C=2048 on 1M features ==="
CUDA_VISIBLE_DEVICES=0 $PY geo1m.py cluster --tag run1m --C 2048 --world 2

echo "=== phase C: extraction on the 32k subset ==="
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py extract_ps --tag run1m --C 2048 \
  --soft_T 1.0 --soft_s 8 --banks_tag prop1m
sz=$(stat -c%s $D/banks_prop1m.pt)
if [ "$sz" -lt 25000000000 ]; then echo "BANKS CHECK FAILED: $sz"; exit 1; fi
echo "banks_prop1m.pt OK ($sz bytes)"

echo "=== phase D: referees (no hard eval per Rohan) ==="
CUDA_VISIBLE_DEVICES=0 $PY german1b.py --tag run1m --banks_tag prop1m \
  --gate_thresh 0.02 --alphas 0.0 -1.0 --space weight &
CUDA_VISIBLE_DEVICES=1 bash -c "
  GEO_DIR=$D $PY emote1b.py prop1m" &
wait
mkdir -p out/full1m
cp $D/*.json out/full1m/ 2>/dev/null || true
cp $D/validate.json out/full1m/ 2>/dev/null || true
echo "=== overnight1m complete ==="
