#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J paired-2k-klkeep
#SBATCH -o logs/paired_2k_klkeep_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_PROJECT=param-autoenc
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto \
  --checkpoint "$PAIR_ROOT/checkpoints/parauto/model.pth" --output "$PAIR_ROOT/output/parauto" --expected-step 2000
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd \
  --checkpoint "$PAIR_ROOT/checkpoints/vpd/model.pth" --output "$PAIR_ROOT/output/vpd" --expected-step 2000
.venv/bin/python -u "$compare/report_milestone.py"
