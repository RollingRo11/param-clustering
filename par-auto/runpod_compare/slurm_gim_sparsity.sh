#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p sharing
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=USR1@90
#SBATCH -J parauto-gim-p025
#SBATCH -o logs/parauto_gim_p025_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 3450 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/gim_p025_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/train_gim_sparsity.py"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
    --checkpoint "$out/checkpoints/005000/model.pth" --sensor gim --run-tag gim-p025-5000-klkeep --output "$out/eval"
.venv/bin/python -u "$compare/report_gim_sparsity.py"
