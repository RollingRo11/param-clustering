#!/bin/bash
cd /workspace/param-clustering/geo-attribution
dev=$1; C=$2; odir=$3; shift 3
for s in "$@"; do
  python3.12 sufficiency67.py --device cuda:$dev --C $C --seed $s --n_samples 48 \
    --out ${odir}/s${s}.json > ${odir}/s${s}.log 2>&1
  echo "done C=$C seed $s"
done
