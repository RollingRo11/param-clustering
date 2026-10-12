#!/bin/bash
# Merge redundancy: how many pieces could be merged exactly or nearly, and what it buys per token.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J parauto-redund
#SBATCH -o logs/parauto_redund_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/redundancy"
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -c "import scipy; print('scipy', scipy.__version__)"
.venv/bin/python -u "$compare/redundancy_probe.py" --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/parauto_1b.json"
.venv/bin/python -u "$compare/redundancy_probe.py" --method vpd --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/vpd_published.json"
.venv/bin/python -u "$compare/redundancy_probe.py" --random-init --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/random_init.json"
echo REDUNDANCY_ALL_DONE
