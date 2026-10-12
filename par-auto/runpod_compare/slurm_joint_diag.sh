#!/bin/bash
# Linear-gap diagnostic on the Fisher-trained and GIM-trained 5k checkpoints.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -J parauto-joint-diag
#SBATCH -o logs/parauto_joint_diag_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/joint_diag"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_joint.py"
.venv/bin/python -u "$compare/joint_diagnostic.py" --checkpoint "$PAIR_ROOT/output/fisher_k4_5000/checkpoints/005000/model.pth" --output "$out/fisher_k4.json"
.venv/bin/python -u "$compare/joint_diagnostic.py" --checkpoint "$PAIR_ROOT/output/gim_1h/checkpoints/005000/model.pth" --output "$out/gim_p05.json"
echo JOINT_DIAG_ALL_DONE
