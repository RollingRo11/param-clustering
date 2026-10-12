#!/bin/bash
# Fisher K=4 score + VPD's usage-density pricing (DENSITY_BETA), 20-step smoke then 5,000 steps.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-density
#SBATCH -o logs/parauto_density_%j.log
set -euo pipefail
export FISHER_K=4 DENSITY_BETA=${DENSITY_BETA:-0.5}
export GIM_DEADLINE=$(( $(date +%s) + 6600 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
smoke="$PAIR_ROOT/output/density_smoke_b${DENSITY_BETA}"
rm -rf "$smoke"
DENSITY_OUT="$smoke" DENSITY_STEPS=20 GIM_DEADLINE=$(( $(date +%s) + 900 )) .venv/bin/python -u "$compare/train_density_5000.py"
echo DENSITY_SMOKE_PASSED
.venv/bin/python -u "$compare/train_density_5000.py"
