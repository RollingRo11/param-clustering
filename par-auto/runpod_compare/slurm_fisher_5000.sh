#!/bin/bash
# Fisher-score training, K=4, 5,000 steps (part 1: profile K=1/2/4/8, then train to the time budget).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-fisher
#SBATCH -o logs/parauto_fisher_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 6600 )) FISHER_K=4
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/smoke_fisher.py" 1 2 4 8
.venv/bin/python -u "$compare/train_fisher_5000.py"
