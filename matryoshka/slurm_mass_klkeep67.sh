#!/bin/bash
# Weight mass of the matryoshka67 comparison checkpoints, then the weight-mass + KLKeep figure.
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -J matryoshka67-mass
#SBATCH -o logs/matryoshka67_mass_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
snap="$PAIR_ROOT/code/matryoshka67_20260923"
out="$PAIR_ROOT/output/matryoshka67_20260923/figures"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/weight_mass67.py" "$out/weight_mass.json"
.venv/bin/python -u "$snap/plot_mass_klkeep67.py" "$out/weight_mass.json" "$out/mass_klkeep.png"
echo MASS_KLKEEP_DONE
