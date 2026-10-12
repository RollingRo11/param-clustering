#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -J parauto-joint-diag-vpd
#SBATCH -o logs/parauto_joint_diag_vpd_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/joint_diagnostic.py" --method vpd --checkpoint "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" --output "$PAIR_ROOT/output/joint_diag/vpd_5000.json"
echo JOINT_DIAG_VPD_DONE
