#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -J parauto-ceiling
#SBATCH -o logs/parauto_ceiling_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/ceiling"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/ceiling_probe.py" --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/parauto_1b.json"
echo CEILING_ALL_DONE
