#!/bin/bash
# Per-piece weight mass (CPU only) of the given weight_mass67.py checkpoint labels.
# Arguments: OUTPUT_NAME LABEL...  -> output/matryoshka67_20260923/figures/OUTPUT_NAME. Needs MATRYOSHKA_SNAP.
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
snap="${MATRYOSHKA_SNAP:?set MATRYOSHKA_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
name="$1"; shift
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/weight_mass67.py" "$PAIR_ROOT/output/matryoshka67_20260923/figures/$name" "$@"
echo MASS_DONE
