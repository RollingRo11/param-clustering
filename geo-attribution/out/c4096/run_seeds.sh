#!/bin/bash
cd /workspace/param-clustering/geo-attribution
dev=$1; tag=$2; pos=$3; wait_for=$4
until [ -f "$wait_for" ]; do sleep 5; done
for s in 1 2 3 4; do
  python3.12 transpose67.py --device cuda:$dev --C 4096 --positions $pos \
    --seed $s --sample_components 256 \
    --out out/c4096/${tag}_fine_s${s}.json > out/c4096/${tag}_fine_s${s}.log 2>&1
  echo "done $tag seed $s -> $(grep -c removable out/c4096/${tag}_fine_s${s}.log) arms"
done
