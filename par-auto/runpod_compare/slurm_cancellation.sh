#!/bin/bash
# Same-mask layer-subset test: does VPD's cross-layer cancellation come from the model or from its masks?
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J parauto-cancel
#SBATCH -o logs/parauto_cancel_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/cancellation"
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
SEQ=${SEQ:-64}
.venv/bin/python -u "$compare/cancellation_probe.py" --sequences $SEQ --method vpd \
    --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/vpd_published.json"
.venv/bin/python -u "$compare/cancellation_probe.py" --sequences $SEQ --sensor fisher \
    --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/parauto_1b.json"
.venv/bin/python -u "$compare/cancellation_probe.py" --sequences $SEQ --method vpd \
    --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" --output "$out/vpd_paired.json"
echo CANCELLATION_ALL_DONE
