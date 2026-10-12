#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p sharing
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J normal-gim-vpd-1k
#SBATCH -o logs/normal_gim_vpd_1k_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/evaluations/gim_normal_vpd_1000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/stage_normal_1000.py"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 1000 \
    --checkpoint "$PAIR_ROOT/checkpoints/normal_1000/model.pth" --sensor gradient \
    --run-tag normal-parauto-1000-klkeep --output "$out/normal"
.venv/bin/python -u "$compare/report_gim_normal_vpd_1000.py"
