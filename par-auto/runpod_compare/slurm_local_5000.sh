#!/bin/bash
# Fisher K=4 score + layer-local sufficiency term (LOCAL_METRIC=avg|raw), 20-step smoke then 5,000 steps.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-local
#SBATCH -o logs/parauto_local_%j.log
set -euo pipefail
export FISHER_K=4 LOCAL_METRIC=${LOCAL_METRIC:-avg} LOCAL_COEFF=${LOCAL_COEFF:-1}
export GIM_DEADLINE=$(( $(date +%s) + 6600 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
smoke="$PAIR_ROOT/output/local_smoke_${LOCAL_METRIC}"
rm -rf "$smoke"
LOCAL_OUT="$smoke" LOCAL_STEPS=20 GIM_DEADLINE=$(( $(date +%s) + 1200 )) .venv/bin/python -u "$compare/train_local_5000.py"
echo LOCAL_SMOKE_PASSED
.venv/bin/python -u "$compare/train_local_5000.py"
