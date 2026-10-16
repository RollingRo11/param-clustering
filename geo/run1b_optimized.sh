#!/bin/bash
# 1B-position streamed decomposition with the validated fast collection setup:
# stat-weighted spec at D=65536 (exact-gram corr 0.95/0.95), bf16 master
# weights (A/B corr 0.9998), batch_seqs=32, direct sparse feature gather.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=${PYTHON:-python3.12}
ARTIFACT_ROOT=${GEO_ATTRIBUTION_ARTIFACT_ROOT:-/dev/shm/geo1b}
export GEO_ATTRIBUTION_ARTIFACT_ROOT="$ARTIFACT_ROOT"
DATA_PATH=$ARTIFACT_ROOT/pile_llama_u32.bin
SPEC_TAG=stream_spec_stats_d65536
PILOT_TAG=run1b_stream_pilot
TAG=run1b_stream
N_POSITIONS=1000000000
PILOT_POSITIONS=131072
POS_PER_SEQ=506
SEQ_LEN=512
BATCH_SEQS=32
MASTER_DTYPE=bfloat16
C=2048
EMBED_DIM=256
RESERVOIR_PER_CLUSTER=16
CHECKPOINT_BATCHES=128

GLOBAL_POSITIONS_PER_BATCH=$((2 * BATCH_SEQS * POS_PER_SEQ))
STREAM_BATCHES=$(((N_POSITIONS + GLOBAL_POSITIONS_PER_BATCH - 1) / GLOBAL_POSITIONS_PER_BATCH))
REQUIRED_DATA_TOKENS=$((STREAM_BATCHES * 2 * BATCH_SEQS * SEQ_LEN))

echo "=== token stream ($REQUIRED_DATA_TOKENS tokens) ==="
AVAILABLE_DATA_TOKENS=0
if [ -f "$DATA_PATH" ]; then
  AVAILABLE_DATA_TOKENS=$(($(stat -c%s "$DATA_PATH") / 4))
fi
if [ "$AVAILABLE_DATA_TOKENS" -lt "$REQUIRED_DATA_TOKENS" ]; then
  "$PY" prep1b.py --target_tokens "$REQUIRED_DATA_TOKENS"
fi

echo "=== bounded pilot fingerprints ($PILOT_POSITIONS positions) ==="
torchrun --nproc_per_node=2 collect_fast.py collect --profile optimized \
  --artifact_root "$ARTIFACT_ROOT" --tag "$PILOT_TAG" --spec_tag "$SPEC_TAG" \
  --data_path "$DATA_PATH" --master_dtype "$MASTER_DTYPE" \
  --n_positions "$PILOT_POSITIONS" --pos_per_seq "$POS_PER_SEQ" \
  --sub_per_seq 0 --batch_seqs "$BATCH_SEQS" --seq_len "$SEQ_LEN" \
  --data_order sequential

echo "=== frozen PCA + spherical centroids (pilot only) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py fit_stream \
  --artifact_root "$ARTIFACT_ROOT" --tag "$TAG" \
  --pilot_tag "$PILOT_TAG" --C "$C" \
  --embed_dim "$EMBED_DIM" --pilot_max_positions "$PILOT_POSITIONS"

echo "=== streamed assignment ($N_POSITIONS positions; resumable) ==="
torchrun --nproc_per_node=2 collect_fast.py stream --profile optimized \
  --artifact_root "$ARTIFACT_ROOT" --tag "$TAG" \
  --spec_tag "$SPEC_TAG" --stream_model_tag "$TAG" \
  --data_path "$DATA_PATH" --master_dtype "$MASTER_DTYPE" \
  --n_positions "$N_POSITIONS" --pos_per_seq "$POS_PER_SEQ" \
  --batch_seqs "$BATCH_SEQS" --seq_len "$SEQ_LEN" --data_order sequential \
  --C "$C" --reservoir_per_cluster "$RESERVOIR_PER_CLUSTER" \
  --checkpoint_batches "$CHECKPOINT_BATCHES" --resume

echo "=== lazy per-module extraction from bounded BF16 reservoirs ==="
CUDA_VISIBLE_DEVICES=0 "$PY" geo1b.py extract_ps --tag "$TAG" --C "$C" \
  --soft_T 1.0 --soft_s 8 --banks_tag prop1b

echo "RUN1B_OPTIMIZED_DONE: $ARTIFACT_ROOT/$TAG"
