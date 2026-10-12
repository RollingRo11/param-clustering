#!/bin/bash
# Decisive test for consumer-gated importance: published VPD at 512 kept, its own CI versus our sensors.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J parauto-consumer
#SBATCH -o logs/parauto_consumer_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/consumer_20260916"
cd "$PAIR_ROOT/code/vpd"
run() { .venv/bin/python -u "$compare/consumer_probe.py" --budgets 512 --sequences 16 --first-row 384 "$@"; }
# GIM rules throughout (sensor and the signed sketch), published VPD: gim vs signed vs its own CI.
run --name vpd_published --sensor gim --modes signed abs --output "$out/vpd_published_gim.json"
# Fisher K=64 as the comparison ranking, matching the 1.533 figure on this slice.
run --name vpd_published --sensor fisher --fisher-k 64 --modes signed --output "$out/vpd_published_fisher64.json"
# Our own factors, same budget.
run --name gim_5000 --sensor gim --modes signed abs --output "$out/gim_5000_gim.json"
echo CONSUMER_DECISIVE_DONE
