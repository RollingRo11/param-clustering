#!/bin/bash
# One shard at a time, task-state-verified: launch, then poll BEAM's task
# state; on ERROR/CANCELLED retry (3x); only trust "blocks" in output.
cd /Users/rohan/Developer/param-clustering/geo-attribution
export PATH="$HOME/.local/bin:$PATH"
U="uv run --with beam-client --python 3.12 python"
for s in 0 1 2 3; do
  ok=""
  for attempt in 1 2 3; do
    out=$($U beam_cofac67.py prestage_shard $s 2>&1 | tail -4)
    if echo "$out" | grep -q '"blocks"'; then
      echo "SHARD_${s}_OK $(echo "$out" | grep -o '"blocks":[^,]*')"
      ok=1; break
    fi
    echo "SHARD_${s}_ATTEMPT_${attempt}_FAILED: $(echo "$out" | tail -1 | head -c 120)"
    sleep 90
  done
  [ -z "$ok" ] && { echo "SHARD_${s}_GAVE_UP"; exit 1; }
done
echo PRESTAGE_V4_DONE
