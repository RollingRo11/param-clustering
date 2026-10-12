#!/bin/bash
# Top held-out contexts per component: par-auto (GIM p=.5, 5k) and VPD (5k).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J parauto-interp
#SBATCH -o logs/parauto_interp_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/interp_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/interp_evidence.py" parauto --sensor gim \
    --checkpoint "$PAIR_ROOT/output/gim_1h/checkpoints/005000/model.pth" --output "$out/parauto_gim_p05.json"
.venv/bin/python -u "$compare/interp_evidence.py" vpd \
    --checkpoint "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" --output "$out/vpd.json"
