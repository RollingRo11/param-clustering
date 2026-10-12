#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -J parauto-perlayer-vf
#SBATCH -o logs/parauto_perlayer_vf_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/perlayer_klkeep.py" --method vpd --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" --output "$PAIR_ROOT/output/perlayer/vpd_paired_final.json"
echo PERLAYER_VF_DONE
