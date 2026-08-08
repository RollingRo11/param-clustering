#!/bin/bash
# Shard the fair sweep across many concurrent processes.
#
# One sweep process is launch-bound, not compute-bound: each step is ~4k tokens
# through a 1B model, which leaves a B200 at ~35% of its power envelope and 19%
# of its memory. Running one (objective, budget) cell per process fills the
# GPUs with independent work instead. Each process still puts the component arm
# on cuda:0 and the LoRA arm on cuda:1, so both cards stay loaded.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON:-python3.12}
mkdir -p out
PIDS=()
for OBJ in english_only multilingual; do
  # Two budgets per shard: 4 shards fit comfortably in memory, where 8 OOM'd.
  for B in "2048 512" "64 8"; do
    NAME="${OBJ}_$(echo $B | tr ' ' '_')"
    OUT="fair_${NAME}.json"
    LOG="out/fair_${NAME}.log"
    "$PY" lora_fair_sweep.py --objectives "$OBJ" --budgets $B \
      --out "$OUT" > "$LOG" 2>&1 &
    PIDS+=($!)
    sleep 3          # stagger model loads; HF from_pretrained is not thread-safe
  done
done
echo "launched ${#PIDS[@]} shards: ${PIDS[*]}"
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL+1)); done
echo "shards finished, $FAIL failures"
"$PY" - <<'EOF'
import json, glob
from pathlib import Path
RUN = Path("/dev/shm/geo1b/run1b_streamC4096")
rows = []
for f in sorted([p for p in sorted(RUN.glob("fair_*.json")) if p.name != "lora_fair_sweep.json"]):
    rows += json.loads(f.read_text())
(RUN / "lora_fair_sweep.json").write_text(json.dumps(rows, indent=2))
print(f"merged {len(rows)} frontier points -> {RUN/'lora_fair_sweep.json'}")
EOF
