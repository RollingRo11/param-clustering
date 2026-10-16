#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J extract-vpd67
#SBATCH -o logs/extract_vpd67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
"$PAIR_ROOT/code/vpd/.venv/bin/python" -u "$PAIR_ROOT/code/compare/extract_vpd_pieces67.py" \
    --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --modules $MODULES_TO_EXTRACT \
    --output "$PAIR_ROOT/output/coactive67/vpd_pieces_${TAG}.pt"
