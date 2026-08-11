#!/bin/bash
# Flagship: Llama-3.2-1B, C=8192, IG K=5, 200M stratified tokens.
# Sensor flags are passed explicitly (NO --profile: apply_profile would
# override sensor/ig_k). Perf flags match the optimized profile.
# Sweep evidence (out/sweep67): 100-200M tokens beats 1B at fixed quota;
# quota 8 (not production's 16) -- narrower reservoir samples were crisper.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3.12
ROOT=/dev/shm/geo1b
CORPUS=/dev/shm/geo1b/pile_llama_u32.bin
export GEO_ATTRIBUTION_ARTIFACT_ROOT="$ROOT"
export GEO_ATTRIBUTION_DATA_PATH="$CORPUS"
mkdir -p "$ROOT"
SENSOR="--sensor ig --ig_k 2 --scalar equal_reward --bf16 --compile --no-fused_attention"
C=32000; N_POS=32000000; PILOT=262144; QUOTA=3
step () { echo; echo "########## $* ##########"; date -Is; }

step "corpus: 260M stratified tokens"
"$PY" prep1b.py --target_tokens 260000000 --text_batch 1024

step "0 bootstrap spec"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py spec \
  --artifact_root "$ROOT" --tag fl32_spec_w2 --feat_dim 65536

step "1 bootstrap pilot"
torchrun --nproc_per_node=2 collect_fast.py collect \
  --artifact_root "$ROOT" --tag fl32_pilot_boot --spec_tag fl32_spec_w2 \
  --data_path "$CORPUS" --master_dtype bfloat16 \
  --n_positions "$PILOT" --pos_per_seq 506 --sub_per_seq 0 \
  --batch_seqs 32 --seq_len 512 --data_order sequential $SENSOR

step "2 bootstrap centroids"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py fit_stream \
  --artifact_root "$ROOT" --tag fl32_stream_boot --pilot_tag fl32_pilot_boot \
  --C "$C" --embed_dim 256 --pilot_max_positions "$PILOT"

step "3 short stream to fill reservoirs"
torchrun --nproc_per_node=2 collect_fast.py stream \
  --artifact_root "$ROOT" --tag fl32_stream_boot --spec_tag fl32_spec_w2 \
  --stream_model_tag fl32_stream_boot --data_path "$CORPUS" \
  --master_dtype bfloat16 --n_positions 4000000 --pos_per_seq 64 \
  --batch_seqs 16 --seq_len 512 --data_order sequential \
  --C "$C" --reservoir_per_cluster 2 --checkpoint_batches 128 \
  --resume $SENSOR

step "4 production spec (stat-weighted)"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py spec_stats \
  --artifact_root "$ROOT" --tag fl32_spec_stats --val_tag fl32_stream_boot \
  --feat_dim 65536 --stats_rows 2048

rm -rf "$ROOT/fl32_stream_boot" "$ROOT/fl32_pilot_boot"
df -h /dev/shm | tail -1
step "5 production pilot"
torchrun --nproc_per_node=2 collect_fast.py collect \
  --artifact_root "$ROOT" --tag fl32_pilot --spec_tag fl32_spec_stats \
  --data_path "$CORPUS" --master_dtype bfloat16 \
  --n_positions "$PILOT" --pos_per_seq 506 --sub_per_seq 0 \
  --batch_seqs 32 --seq_len 512 --data_order sequential $SENSOR

step "6 frozen PCA + spherical centroids C=$C"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py fit_stream \
  --artifact_root "$ROOT" --tag fl32_streamC8192 --pilot_tag fl32_pilot \
  --C "$C" --embed_dim 256 --pilot_max_positions "$PILOT"

step "7 main stream: $N_POS positions"
torchrun --nproc_per_node=2 collect_fast.py stream \
  --artifact_root "$ROOT" --tag fl32_streamC8192 --spec_tag fl32_spec_stats \
  --stream_model_tag fl32_streamC8192 --data_path "$CORPUS" \
  --master_dtype bfloat16 --n_positions "$N_POS" --pos_per_seq 64 \
  --batch_seqs 16 --seq_len 512 --data_order sequential \
  --C "$C" --reservoir_per_cluster "$QUOTA" --checkpoint_batches 128 \
  --resume $SENSOR

rm -rf "$ROOT/fl32_pilot"
df -h /dev/shm | tail -1
step "8 ownership extraction -> banks"
CUDA_VISIBLE_DEVICES=0 "$PY" geo1b.py extract_ps --tag fl32_streamC8192 \
  --C "$C" --soft_T 1.0 --soft_s 8 --banks_tag flagship

echo "FLAGSHIP_DONE: $ROOT/fl32_streamC8192"
date -Is
