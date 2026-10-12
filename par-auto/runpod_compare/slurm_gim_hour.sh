#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p sharing
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=USR1@60
#SBATCH -J parauto-gim-1h
#SBATCH -o logs/parauto_gim_1h_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 3540 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_gim_attribution.py"
.venv/bin/python -u "$compare/train_gim_hour.py"
