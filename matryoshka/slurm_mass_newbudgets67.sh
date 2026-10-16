#!/bin/bash
# Per-piece weight mass of the lambda = 1 and new-budget Matryoshka 5k checkpoints (CPU only), next to the cached
# figures/weight_mass.json of the earlier runs and both VPD checkpoints.
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J matryoshka67-mass
#SBATCH -o logs/matryoshka67_mass_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
snap="$PAIR_ROOT/code/matryoshka67_mass_20261002"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/weight_mass67.py" "$PAIR_ROOT/output/matryoshka67_20260923/figures/weight_mass_lp1_newbudgets.json" \
    matryoshka_lp1_5000 matryoshka_lp1_newbudgets_5000
echo MASS_DONE
