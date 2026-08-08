#!/bin/bash
cd /workspace/param-clustering/geo-attribution
dev=$1; shift
for s in "$@"; do
  python3.12 sensor_study67.py --device cuda:$dev --C 4096 --seed $s \
    --sample_components 256 --out out/sensors/c4096_s${s}.json \
    > out/sensors/c4096_s${s}.log 2>&1
  echo "done c4096 seed $s"
done
