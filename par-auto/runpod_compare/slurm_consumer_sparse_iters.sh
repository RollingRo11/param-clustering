#!/bin/bash
# Control for the sparse-point sweep: the on-set never settled at 3 rounds, so score once (1 round) and long (6 rounds).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH -t 00:45:00
#SBATCH -J parauto-consumer-iters
#SBATCH -o logs/parauto_consumer_iters_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/consumer_sparse_20260916"
cd "$PAIR_ROOT/code/vpd"
run() { .venv/bin/python -u "$compare/consumer_probe.py" --budgets 512 --sequences 16 --first-row 384 --modes signed sparse "$@"; }
for it in 1 6; do
  run --name vpd_published --sensor fisher --fisher-k 64 --sketch-k 64 --iterations $it --output "$out/vpd_published_matched64_it$it.json"
  run --name gim_5000     --sensor fisher --fisher-k 64 --sketch-k 64 --iterations $it --output "$out/gim_5000_matched64_it$it.json"
  run --name vpd_published --sensor gim --iterations $it --output "$out/vpd_published_gim_it$it.json"
done
echo CONSUMER_ITERS_DONE
