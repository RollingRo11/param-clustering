#!/bin/bash
cd /workspace/param-clustering/geo-attribution
dev=$1; shift
for s in "$@"; do
  python3.12 sufficiency67.py --device cuda:$dev --C 256 --seed $s --n_samples 48 \
    --out out/suff/s${s}.json > out/suff/s${s}.log 2>&1
  echo "done suff seed $s"
done
