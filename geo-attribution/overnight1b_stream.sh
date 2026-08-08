#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=${PYTHON:-python}
ARTIFACT_ROOT=${GEO_ATTRIBUTION_ARTIFACT_ROOT:-/dev/shm/geo1b}
export GEO_ATTRIBUTION_ARTIFACT_ROOT="$ARTIFACT_ROOT"
DATA_PATH=${GEO_ATTRIBUTION_DATA_PATH:-$ARTIFACT_ROOT/pile_llama_u32.bin}
SPEC_TAG=${SPEC_TAG:-stream_spec}
FEAT_DIM=${FEAT_DIM:-16384}
PILOT_TAG=${PILOT_TAG:-run1b_pilot}
TAG=${TAG:-run1b}
N_POSITIONS=${N_POSITIONS:-1000000000}
PILOT_POSITIONS=${PILOT_POSITIONS:-131072}
POS_PER_SEQ=${POS_PER_SEQ:-506}
SEQ_LEN=${SEQ_LEN:-512}
BATCH_SEQS=${BATCH_SEQS:-4}
C=${C:-2048}
EMBED_DIM=${EMBED_DIM:-256}
RESERVOIR_PER_CLUSTER=${RESERVOIR_PER_CLUSTER:-16}
CHECKPOINT_BATCHES=${CHECKPOINT_BATCHES:-128}

GLOBAL_POSITIONS_PER_BATCH=$((2 * BATCH_SEQS * POS_PER_SEQ))
STREAM_BATCHES=$(((N_POSITIONS + GLOBAL_POSITIONS_PER_BATCH - 1) / GLOBAL_POSITIONS_PER_BATCH))
REQUIRED_DATA_TOKENS=$((STREAM_BATCHES * 2 * BATCH_SEQS * SEQ_LEN))
if [ ! -f "$DATA_PATH" ]; then
  echo "missing token stream: $DATA_PATH"
  echo "run: $PY prep1b.py --target_tokens $REQUIRED_DATA_TOKENS"
  exit 1
fi
AVAILABLE_DATA_TOKENS=$(($(stat -c%s "$DATA_PATH") / 4))
if [ "$AVAILABLE_DATA_TOKENS" -lt "$REQUIRED_DATA_TOKENS" ]; then
  echo "token stream has $AVAILABLE_DATA_TOKENS tokens; $REQUIRED_DATA_TOKENS required"
  echo "run: $PY prep1b.py --target_tokens $REQUIRED_DATA_TOKENS"
  exit 1
fi

echo "=== reusable attribution feature spec ==="
if [ ! -f "$ARTIFACT_ROOT/$SPEC_TAG/spec.pt" ]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py spec \
    --artifact_root "$ARTIFACT_ROOT" --tag "$SPEC_TAG" --feat_dim "$FEAT_DIM"
fi

echo "=== bounded pilot fingerprints ($PILOT_POSITIONS positions) ==="
torchrun --nproc_per_node=2 collect_fast.py collect --profile optimized \
  --artifact_root "$ARTIFACT_ROOT" --tag "$PILOT_TAG" --spec_tag "$SPEC_TAG" \
  --data_path "$DATA_PATH" \
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
  --data_path "$DATA_PATH" \
  --n_positions "$N_POSITIONS" --pos_per_seq "$POS_PER_SEQ" \
  --batch_seqs "$BATCH_SEQS" --seq_len "$SEQ_LEN" --data_order sequential \
  --C "$C" --reservoir_per_cluster "$RESERVOIR_PER_CLUSTER" \
  --checkpoint_batches "$CHECKPOINT_BATCHES" --resume

echo "=== lazy per-module extraction from bounded BF16 reservoirs ==="
CUDA_VISIBLE_DEVICES=0 "$PY" geo1b.py extract_ps --tag "$TAG" --C "$C" \
  --soft_T 1.0 --soft_s 8 --banks_tag prop1b

echo "=== streamed 1B decomposition complete: $ARTIFACT_ROOT/$TAG ==="
