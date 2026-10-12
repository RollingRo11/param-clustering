#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -J gim-vpd-1k-stage
#SBATCH -o logs/gim_vpd_1k_stage_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$PAIR_ROOT/code/compare/stage_gim_vpd_1000.py"
