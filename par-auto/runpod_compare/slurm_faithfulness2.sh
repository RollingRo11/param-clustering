#!/bin/bash
# Controls: all components kept (is the deviation from masking at all?) and untrained random pieces.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J parauto-faithful2
#SBATCH -o logs/parauto_faithful2_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/faithfulness"
ours="$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/faithfulness_probe.py" --keep 1.0 --sensor fisher --checkpoint "$ours" --output "$out/parauto_1b_keep1.0.json"
.venv/bin/python -u "$compare/faithfulness_probe.py" --keep 1.0 --method vpd --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/vpd_published_keep1.0.json"
.venv/bin/python -u "$compare/faithfulness_probe.py" --keep 1.0 --method vpd --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" --output "$out/vpd_paired_keep1.0.json"
.venv/bin/python -u "$compare/faithfulness_probe.py" --keep 0.01 --sensor fisher --random-init --checkpoint "$ours" --output "$out/random_init_keep0.01.json"
echo FAITHFULNESS2_ALL_DONE
