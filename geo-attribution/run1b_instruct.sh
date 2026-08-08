#!/bin/bash
# Same decomposition pipeline as run1b_c4096.sh, retargeted at
# Llama-3.2-1B-INSTRUCT.
#
# The stat-weighted feature proposal is model-specific (q ∝ W² · E[g²] · E[p²])
# and is built FROM reservoir rows, which only exist after a stream has run —
# so the good spec has to be bootstrapped rather than copied from the base
# model's run:
#
#   0  weak W²-only spec                (no data needed)
#   1  pilot with it                    -> fingerprints
#   2  fit PCA + spherical centroids    -> throwaway stream model
#   3  short stream                     -> fills reservoirs with (p, g) rows
#   4  spec_stats off those reservoirs  -> the production D=65536 spec
#   5  pilot again with the good spec
#   6  fit PCA + spherical centroids    -> the frozen decomposition
#   7  full 1B-position streamed assignment (resumable)
#   8  extract per-module ownership     -> banks
#
# Artifacts go to real disk (/workspace), not tmpfs: the base-model runs
# already occupy 316G of the 352G /dev/shm, and disk survives a reboot.
# The token corpus stays in tmpfs because it is the read-hot path.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=${PYTHON:-python3.12}
export GEO_MODEL_ID=${GEO_MODEL_ID:-unsloth/Llama-3.2-1B-Instruct}
export GEO_MODEL_REVISION=${GEO_MODEL_REVISION:-5a8abab4a5d6f164389b1079fb721cfab8d7126c}

ARTIFACT_ROOT=${GEO_ATTRIBUTION_ARTIFACT_ROOT:-/workspace/geo1b_instruct}
export GEO_ATTRIBUTION_ARTIFACT_ROOT="$ARTIFACT_ROOT"
mkdir -p "$ARTIFACT_ROOT"
DATA_PATH=${GEO_DATA_PATH:-/dev/shm/geo1b/pile_llama_u32.bin}

SPEC_W2=instruct_spec_w2
BOOT_PILOT=instruct_pilot_boot
BOOT_STREAM=instruct_stream_boot
SPEC_TAG=instruct_spec_stats_d65536
PILOT_TAG=instruct_pilot262k
TAG=instruct_streamC4096

N_POSITIONS=${N_POSITIONS:-1000000000}
BOOT_POSITIONS=${BOOT_POSITIONS:-20000000}
PILOT_POSITIONS=262144
POS_PER_SEQ=506
SEQ_LEN=512
BATCH_SEQS=32
MASTER_DTYPE=bfloat16
FEAT_DIM=65536
C=4096
EMBED_DIM=256
RESERVOIR_PER_CLUSTER=16
CHECKPOINT_BATCHES=128
STATS_ROWS=${STATS_ROWS:-2048}

echo "=== model: $GEO_MODEL_ID @ $GEO_MODEL_REVISION"
echo "=== artifacts: $ARTIFACT_ROOT   corpus: $DATA_PATH"
df -h "$ARTIFACT_ROOT" | tail -1

step () { echo; echo "########## $* ##########"; date -Is; }

step "0/8 bootstrap spec (W^2 proposal, D=$FEAT_DIM)"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py spec \
  --artifact_root "$ARTIFACT_ROOT" --tag "$SPEC_W2" --feat_dim "$FEAT_DIM"

step "1/8 bootstrap pilot ($PILOT_POSITIONS positions)"
torchrun --nproc_per_node=2 collect_fast.py collect --profile optimized \
  --artifact_root "$ARTIFACT_ROOT" --tag "$BOOT_PILOT" --spec_tag "$SPEC_W2" \
  --data_path "$DATA_PATH" --master_dtype "$MASTER_DTYPE" \
  --n_positions "$PILOT_POSITIONS" --pos_per_seq "$POS_PER_SEQ" \
  --sub_per_seq 0 --batch_seqs "$BATCH_SEQS" --seq_len "$SEQ_LEN" \
  --data_order sequential

step "2/8 bootstrap centroids"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py fit_stream \
  --artifact_root "$ARTIFACT_ROOT" --tag "$BOOT_STREAM" \
  --pilot_tag "$BOOT_PILOT" --C "$C" --embed_dim "$EMBED_DIM" \
  --pilot_max_positions "$PILOT_POSITIONS"

step "3/8 short stream ($BOOT_POSITIONS positions) to fill reservoirs"
torchrun --nproc_per_node=2 collect_fast.py stream --profile optimized \
  --artifact_root "$ARTIFACT_ROOT" --tag "$BOOT_STREAM" \
  --spec_tag "$SPEC_W2" --stream_model_tag "$BOOT_STREAM" \
  --data_path "$DATA_PATH" --master_dtype "$MASTER_DTYPE" \
  --n_positions "$BOOT_POSITIONS" --pos_per_seq "$POS_PER_SEQ" \
  --batch_seqs "$BATCH_SEQS" --seq_len "$SEQ_LEN" --data_order sequential \
  --C "$C" --reservoir_per_cluster "$RESERVOIR_PER_CLUSTER" \
  --checkpoint_batches "$CHECKPOINT_BATCHES" --resume

step "4/8 production spec (stat-weighted, D=$FEAT_DIM)"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py spec_stats \
  --artifact_root "$ARTIFACT_ROOT" --tag "$SPEC_TAG" \
  --val_tag "$BOOT_STREAM" --feat_dim "$FEAT_DIM" --stats_rows "$STATS_ROWS"

step "5/8 production pilot ($PILOT_POSITIONS positions)"
torchrun --nproc_per_node=2 collect_fast.py collect --profile optimized \
  --artifact_root "$ARTIFACT_ROOT" --tag "$PILOT_TAG" --spec_tag "$SPEC_TAG" \
  --data_path "$DATA_PATH" --master_dtype "$MASTER_DTYPE" \
  --n_positions "$PILOT_POSITIONS" --pos_per_seq "$POS_PER_SEQ" \
  --sub_per_seq 0 --batch_seqs "$BATCH_SEQS" --seq_len "$SEQ_LEN" \
  --data_order sequential

step "6/8 frozen PCA + spherical centroids (C=$C)"
CUDA_VISIBLE_DEVICES=0 "$PY" collect_fast.py fit_stream \
  --artifact_root "$ARTIFACT_ROOT" --tag "$TAG" \
  --pilot_tag "$PILOT_TAG" --C "$C" --embed_dim "$EMBED_DIM" \
  --pilot_max_positions "$PILOT_POSITIONS"

step "7/8 streamed assignment ($N_POSITIONS positions; resumable)"
torchrun --nproc_per_node=2 collect_fast.py stream --profile optimized \
  --artifact_root "$ARTIFACT_ROOT" --tag "$TAG" \
  --spec_tag "$SPEC_TAG" --stream_model_tag "$TAG" \
  --data_path "$DATA_PATH" --master_dtype "$MASTER_DTYPE" \
  --n_positions "$N_POSITIONS" --pos_per_seq "$POS_PER_SEQ" \
  --batch_seqs "$BATCH_SEQS" --seq_len "$SEQ_LEN" --data_order sequential \
  --C "$C" --reservoir_per_cluster "$RESERVOIR_PER_CLUSTER" \
  --checkpoint_batches "$CHECKPOINT_BATCHES" --resume

step "8/8 per-module ownership extraction -> banks"
CUDA_VISIBLE_DEVICES=0 "$PY" geo1b.py extract_ps --tag "$TAG" --C "$C" \
  --soft_T 1.0 --soft_s 8 --banks_tag prop1b

echo
echo "RUN1B_INSTRUCT_DONE: $ARTIFACT_ROOT/$TAG"
date -Is
