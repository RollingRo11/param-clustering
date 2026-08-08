#!/bin/bash
cd /workspace/param-clustering/geo-attribution
dev=$1; shift
for s in "$@"; do
  python3.12 combine67.py --device cuda:$dev --seed $s --n_samples 48 \
    --base gim --sensors gim,eap --out out/comb/s${s}.json > out/comb/s${s}.log 2>&1
  echo "done comb seed $s"
done
