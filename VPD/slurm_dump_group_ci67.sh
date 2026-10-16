#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J dump-group-ci67
#SBATCH -o logs/dump_group_ci67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/coactive67"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/dump_group_ci67.py" --device cpu --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" \
    --groups "$out/vpd_published.json" --rows "${ROWS:-256}" --output "$out/group_ci${SUFFIX:-}.pt"
