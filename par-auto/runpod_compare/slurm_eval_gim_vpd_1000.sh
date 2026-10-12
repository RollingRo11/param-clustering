#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p sharing
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J gim-vpd-1k-klkeep
#SBATCH -o logs/gim_vpd_1k_klkeep_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/evaluations/gim_vpd_1000"
checkpoints="$PAIR_ROOT/checkpoints/gim_vpd_1000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 1000 \
  --checkpoint "$PAIR_ROOT/output/gim_1h/checkpoints/001000/model.pth" --sensor gim --run-tag gim-1000-klkeep --output "$out/parauto"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 1000 \
  --checkpoint "$checkpoints/vpd.pth" --run-tag vpd-1000-klkeep --output "$out/vpd"
.venv/bin/python -u "$compare/report_gim_vpd_1000.py" --step 1000 --output "$out"
