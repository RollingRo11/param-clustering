#!/bin/bash
# Layer-wise / attn-vs-mlp masking, exact layer-0 output change, VPD-by-Fisher ranking.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -J parauto-gap-probe
#SBATCH -o logs/parauto_gap_probe_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/gap_probe"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/gap_probe.py" --method vpd --checkpoint "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" --output "$out/vpd_5000.json"
.venv/bin/python -u "$compare/gap_probe.py" --checkpoint "$PAIR_ROOT/output/fisher_k4_5000/checkpoints/005000/model.pth" --output "$out/fisher_k4.json"
echo GAP_PROBE_ALL_DONE
