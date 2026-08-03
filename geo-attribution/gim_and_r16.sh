#!/bin/bash
set -e
cd /workspace/circuit-decomp/geo-attribution
PY=/usr/bin/python3.12

echo "=== GIM arm: collect+gram (DDP) ==="
$PY -m torch.distributed.run --standalone --nproc_per_node=2 geo67.py collect \
  --tag gim32 --sensor gim --n_positions 32768 --pos_per_seq 16 --batch_seqs 16 --seq_len 512 || true
$PY - <<'PYEOF'
import torch
for r in (0,1):
    d = torch.load(f"out/gim32/collect_rank{r}.pt", weights_only=True, map_location="cpu")
    assert d["n"] == 16384 and d["sensor"] == "gim", (r, d["n"])
print("collect shards verified")
PYEOF
sleep 30
$PY -m torch.distributed.run --standalone --nproc_per_node=2 geo67.py gram --tag gim32 --gram_block 256 || true
test -f out/gim32/gram.pt
CUDA_VISIBLE_DEVICES=0 $PY geo67.py factor --tag gim32 --C 512
CUDA_VISIBLE_DEVICES=0 $PY geo67.py extract_p --tag gim32 --C 512 --base_ratio 8 --banks_tag gim_r8 &
CUDA_VISIBLE_DEVICES=1 $PY geo67.py extract_p --tag gim32 --C 512 --base_ratio 5 --banks_tag gim_r5 &
wait
echo "=== GIM battery + full65 r12/r16 arms ==="
CUDA_VISIBLE_DEVICES=0 bash -c "
  $PY geo67.py eval   --tag gim32 --banks_tag gim_r8 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py canary --tag gim32 --banks_tag gim_r8 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256 &&
  $PY german67.py --tag gim32 --banks_tag gim_r8 --gate_thresh 0.02 --alphas 0.0 -1.0 -2.0 &&
  $PY geo67.py extract_p --tag full65 --C 512 --base_ratio 16 --banks_tag C512_r16 &&
  $PY geo67.py eval   --tag full65 --banks_tag C512_r16 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py canary --tag full65 --banks_tag C512_r16 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256 &&
  $PY german67.py --tag full65 --banks_tag C512_r16 --gate_thresh 0.02 --alphas 0.0 -1.0 -2.0" &
CUDA_VISIBLE_DEVICES=1 bash -c "
  $PY geo67.py eval   --tag gim32 --banks_tag gim_r5 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py canary --tag gim32 --banks_tag gim_r5 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256 &&
  $PY geo67.py extract_p --tag full65 --C 512 --base_ratio 12 --banks_tag C512_r12 &&
  $PY geo67.py eval   --tag full65 --banks_tag C512_r12 --gate_thresh 0.02 --eval_batches 6 --batch_seqs 8 --seq_len 512 &&
  $PY geo67.py canary --tag full65 --banks_tag C512_r12 --gate_thresh 0.02 --batch_seqs 16 --seq_len 256 &&
  $PY german67.py --tag full65 --banks_tag C512_r12 --gate_thresh 0.02 --alphas 0.0 -1.0 -2.0" &
wait
echo "=== all arms complete ==="
