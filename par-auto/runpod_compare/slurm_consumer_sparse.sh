#!/bin/bash
# Same sweep as consumer_decisive / rest / matched, plus the sketch re-cached at the sparse point.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 01:30:00
#SBATCH -J parauto-consumer-sparse
#SBATCH -o logs/parauto_consumer_sparse_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/consumer_sparse_20260916"
cd "$PAIR_ROOT/code/vpd"
run() { .venv/bin/python -u "$compare/consumer_probe.py" --budgets 512 --sequences 16 --first-row 384 --modes signed sparse "$@"; }
# GIM rules: one true-token probe, intact-point sketch vs sparse-point sketch vs the base sensor (and VPD's own CI).
run --name vpd_published --sensor gim --output "$out/vpd_published_gim.json"
run --name gim_5000     --sensor gim --output "$out/gim_5000_gim.json"
# Probe-matched exact rules: 64 sampled labels for the flat Fisher sensor and both sketches.
run --name vpd_published --sensor fisher --fisher-k 64 --sketch-k 64 --output "$out/vpd_published_matched64.json"
run --name gim_5000     --sensor fisher --fisher-k 64 --sketch-k 64 --output "$out/gim_5000_matched64.json"
echo CONSUMER_SPARSE_DONE
