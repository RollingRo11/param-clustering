#!/bin/bash
# Checks (sped-up update vs the one-micro-batch-at-a-time reference), then a timing breakdown of one Matryoshka
# training step (lambda = 1, hard forward, Fisher score) on an H200.
# Writes output/matryoshka67_20260923/profile/profile_h200_fast.json; trains nothing and saves no parameters.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -J matryoshka67-profile
#SBATCH -o logs/matryoshka67_profile_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MATRYOSHKA_FORWARD=hard MATRYOSHKA_SENSOR=fisher MATRYOSHKA_LP_COEFF=1 MATRYOSHKA_CHUNK_CONTEXTS=16 WANDB_MODE=disabled
snap="$PAIR_ROOT/code/matryoshka67_20260927"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/test_matryoshka67.py"
.venv/bin/python -u "$snap/profile_matryoshka67.py" "$PAIR_ROOT/output/matryoshka67_20260923/profile/profile_h200_fast.json"
