#!/bin/bash
# Finite-difference check of the signed sketch at a SPARSE on-set, against the masked forward.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -J parauto-sparse-verify
#SBATCH -o logs/parauto_sparse_verify_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
run() { .venv/bin/python -u "$compare/consumer_probe.py" --verify-sparse --first-row 384 --fisher-k 8 --verify-budget 512 --output /dev/null "$@"; }
echo "=== gim_5000 ==="; run --name gim_5000
echo "=== vpd_published ==="; run --name vpd_published
echo SPARSE_VERIFY_JOB_DONE
