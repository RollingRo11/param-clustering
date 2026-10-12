#!/bin/bash
# Same-set comparison on matryoshka67's held-out monitor (val rows 1000-1003, fixed budgets, native rankings):
# VPD at 1,000 and 5,000 steps, the fisher_k4_5000 control at 1,000 and 5,000, and both Matryoshka runs at 1,000.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J matryoshka67-compare
#SBATCH -o logs/matryoshka67_compare_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
snap="$PAIR_ROOT/code/matryoshka67_20260923"
exp="$PAIR_ROOT/output/matryoshka67_20260923"
out="$exp/monitor_compare"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
py=.venv/bin/python
$py -u "$snap/vpd_monitor67.py" "$PAIR_ROOT/checkpoints/gim_vpd_1000/vpd.pth" "$out/vpd_1000.json"
$py -u "$snap/vpd_monitor67.py" "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" "$out/vpd_5000.json"
$py -u "$snap/parauto_monitor67.py" "$PAIR_ROOT/output/fisher_k4_5000/checkpoints/001000/model.pth" "$out/fisher_k4_1000.json"
$py -u "$snap/parauto_monitor67.py" "$PAIR_ROOT/output/fisher_k4_5000/checkpoints/005000/model.pth" "$out/fisher_k4_5000.json"
$py -u "$snap/parauto_monitor67.py" "$exp/main/checkpoints/001000/model.pth" "$out/matryoshka_soft_1000.json"
$py -u "$snap/parauto_monitor67.py" "$exp/main_hard/checkpoints/001000/model.pth" "$out/matryoshka_hard_1000.json"
echo MONITOR_COMPARE_DONE
