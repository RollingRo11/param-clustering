#!/bin/bash
# VPD's first 50 steps (paired harness and published config, as the paired 1B run), then matryoshka67's
# held-out monitor on that checkpoint, for a step-matched comparison with the Matryoshka pilot at step 50.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J matryoshka67-vpd50
#SBATCH -o logs/matryoshka67_vpd50_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
snap="$PAIR_ROOT/code/matryoshka67_20260923"
out="$PAIR_ROOT/output/matryoshka67_20260923/vpd_step50"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
# train_pair.py writes pilots to output/pilot/vpd; refuse to touch an existing one.
test ! -e "$PAIR_ROOT/output/pilot/vpd"
test ! -e "$out"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$PAIR_ROOT/code/compare/train_pair.py" vpd --pilot-steps 50 --checkpoint-interval 50
mv "$PAIR_ROOT/output/pilot/vpd" "$out"
rmdir "$PAIR_ROOT/output/pilot" 2>/dev/null || true
.venv/bin/python -u "$snap/vpd_monitor67.py" "$out/final.pth" "$out/monitor.json"
echo VPD_STEP50_DONE
