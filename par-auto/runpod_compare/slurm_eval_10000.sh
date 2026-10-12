#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J paired-10k-klkeep
#SBATCH -o logs/paired_10k_klkeep_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/evaluations/10000"
checkpoints="$PAIR_ROOT/checkpoints/step_10000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 10000 \
  --checkpoint "$checkpoints/parauto.pth" --output "$out/parauto"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 10000 \
  --checkpoint "$checkpoints/vpd.pth" --output "$out/vpd"
.venv/bin/python -u "$compare/report_milestone.py" --step 10000 --output "$out"
