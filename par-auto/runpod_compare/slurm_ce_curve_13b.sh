#!/bin/bash
# Add the stopped 13B GIM run (last checkpoint, step 36,000) to the CE/KL curve; same settings as slurm_ce_curve.sh.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J parauto-ce-curve-13b
#SBATCH -o logs/parauto_ce_curve_13b_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/ce_curve_20260916"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 36000 --sensor fisher --fisher-k 8 \
    --checkpoint "$PAIR_ROOT/output/gim_13b/checkpoints/036000/model.pth" \
    --run-tag ce-curve-parauto-gim-13b-36k --output "$out/parauto_gim_13b_36k"
echo CE_CURVE_13B_DONE
