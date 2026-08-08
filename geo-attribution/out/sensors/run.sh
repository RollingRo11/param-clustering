#!/bin/bash
cd /workspace/param-clustering/geo-attribution
dev=$1; shift
for s in "$@"; do
  python3.12 sensor_study67.py --device cuda:$dev --C 256 --seed $s \
    --out out/sensors/c256_s${s}.json > out/sensors/c256_s${s}.log 2>&1
  echo "done c256 seed $s"
done
