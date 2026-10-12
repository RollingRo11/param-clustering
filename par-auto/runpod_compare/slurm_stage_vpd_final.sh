#!/bin/bash
# Fetch the latest retained paired-VPD snapshot from R2 (20k if present, else 15k).
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -J vpd-final-stage
#SBATCH -o logs/vpd_final_stage_%j.log
set -uo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$PAIR_ROOT/code/compare/stage_vpd_final.py"
