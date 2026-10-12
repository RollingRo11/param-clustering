#!/bin/bash
# Part 1 of 2: train until the time budget, checkpointing every 500 updates.
# Part 2 (slurm_finite_5000_finish.sh, afterany) resumes to 5,000 and evaluates.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-finite
#SBATCH -o logs/parauto_finite_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 6900 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_finite_ablation.py"
.venv/bin/python -u "$compare/train_finite_5000.py"
