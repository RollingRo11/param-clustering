#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -J parauto-joint-diag-j
#SBATCH -o logs/parauto_joint_diag_joint_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/joint_diagnostic.py" --checkpoint "$PAIR_ROOT/output/fisher_k4_joint_5000/checkpoints/005000/model.pth" --output "$PAIR_ROOT/output/joint_diag/fisher_k4_joint.json"
echo JOINT_DIAG_JOINT_DONE
