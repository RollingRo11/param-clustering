#!/bin/bash
# CE degradation from baseline (and KL) on the same 128 held-out sequences, five checkpoints.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J parauto-ce-curve
#SBATCH -o logs/parauto_ce_curve_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/ce_curve_20260916"
cd "$PAIR_ROOT/code/vpd"
run() { .venv/bin/python -u "$compare/evaluate_checkpoint.py" "$@"; }
run parauto --expected-step 30518 --sensor fisher --fisher-k 8 \
    --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" \
    --run-tag ce-curve-parauto-1b --output "$out/parauto_1b"
run parauto --expected-step 5000 --sensor fisher --fisher-k 8 \
    --checkpoint "$PAIR_ROOT/output/gim_1h/checkpoints/005000/model.pth" \
    --run-tag ce-curve-parauto-gim-5k --output "$out/parauto_gim_5k"
run vpd --expected-step 5000 \
    --checkpoint "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" \
    --run-tag ce-curve-vpd-5k --output "$out/vpd_5k"
run vpd --expected-step 18664 \
    --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" \
    --run-tag ce-curve-vpd-0.6b --output "$out/vpd_0.6b"
run vpd --expected-step 400000 \
    --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" \
    --run-tag ce-curve-vpd-13b --output "$out/vpd_13b"
echo CE_CURVE_DONE
