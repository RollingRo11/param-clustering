#!/bin/bash
# Upstream damage x downstream mode grid: is repair a model behaviour or a masking artifact?
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J parauto-repair
#SBATCH -o logs/parauto_repair_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/repair"
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
VPD=${VPD:-published}
case $VPD in published) ck="$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth";; paired) ck="$PAIR_ROOT/checkpoints/vpd_final/vpd.pth";; esac
.venv/bin/python -u "$compare/repair_probe.py" --sequences ${SEQ:-64} --vpd "$ck" \
    --parauto "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/vpd_${VPD}.json"
echo REPAIR_DONE
