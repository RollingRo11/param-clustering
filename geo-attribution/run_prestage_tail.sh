#!/bin/bash
cd /Users/rohan/Developer/param-clustering/geo-attribution
U="uv run --with beam-client --python 3.12 python"
# wait for the first wave to end
while ! grep -qa "ALL_SHARDS_ENDED" /private/tmp/claude-501/-Users-rohan-Developer-param-clustering/30021c40-d2f5-4c1b-89c6-097ef77996b9/tasks/bxh8v8hjv.output 2>/dev/null; do sleep 120; done
# relaunch any shard that never reported blocks
for s in 0 1 2 3; do
  if ! grep -qa "\"shard\": $s" /private/tmp/claude-501/-Users-rohan-Developer-param-clustering/30021c40-d2f5-4c1b-89c6-097ef77996b9/tasks/bxh8v8hjv.output; then
    echo "relaunching shard $s"
    for attempt in 1 2 3; do
      $U beam_cofac67.py prestage_shard $s && break
      sleep 120
    done &
    # at most 2 in flight
    while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 30; done
  fi
done
wait
echo TAIL_SHARDS_DONE
