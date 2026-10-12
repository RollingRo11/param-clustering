#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J published-vs-parauto-klkeep
#SBATCH -o logs/published_klkeep_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/published_vpd_vs_finished_parauto"
checkpoints="$PAIR_ROOT/checkpoints/published_vs_finished"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 30518 \
  --checkpoint "$checkpoints/parauto.pth" --output "$out/parauto" --run-tag finished-parauto-klkeep
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 400000 \
  --checkpoint "$checkpoints/vpd.pth" --output "$out/vpd" --run-tag published-vpd-klkeep
.venv/bin/python -u "$compare/report_published_comparison.py"
