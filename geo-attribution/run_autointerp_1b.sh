#!/bin/bash
# Full 1B-position evidence sweep on both GPUs, then merge.
# 61,760 batches/rank x 16 seqs x 506 positions x 2 ranks = 1.00B positions.
set -e
cd "$(dirname "$0")"
B=${1:-61760}
python3.12 autointerp_sweep.py sweep --rank 0 --world 2 --device cuda:0 --batches "$B" --pos_per_seq 506 > out/sweep1b_r0.log 2>&1 &
P0=$!
python3.12 autointerp_sweep.py sweep --rank 1 --world 2 --device cuda:1 --batches "$B" --pos_per_seq 506 > out/sweep1b_r1.log 2>&1 &
P1=$!
wait $P0 $P1
python3.12 autointerp_sweep.py merge --world 2 --out evidence_prop1b_1B.json > out/sweep1b_merge.log 2>&1
tail -1 out/sweep1b_merge.log
