#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J coactive-vpd67
#SBATCH -o logs/coactive_vpd67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/coactive_vpd67.py" --device cpu --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" \
    --rows "${ROWS:-256}" --output "$PAIR_ROOT/output/coactive67/vpd_published${SUFFIX:-}.json"
