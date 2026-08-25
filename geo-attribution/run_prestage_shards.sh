#!/bin/bash
cd /Users/rohan/Developer/param-clustering/geo-attribution
U="uv run --with beam-client --python 3.12 python"
for s in 0 1 2 3; do
  (for attempt in 1 2; do
     $U beam_cofac67.py prestage_shard $s && break
     echo "shard $s attempt $attempt failed; retrying in 60s"
     sleep 60
   done
   echo "SHARD_${s}_ENDED") &
done
wait
echo ALL_SHARDS_ENDED
