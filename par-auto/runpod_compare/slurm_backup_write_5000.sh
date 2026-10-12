#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J parauto-write-backup
#SBATCH -o logs/parauto_write_backup_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$PAIR_ROOT/code/compare/backup_write_5000.py"
