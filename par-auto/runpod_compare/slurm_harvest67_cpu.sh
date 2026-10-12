#!/bin/bash
# Max-activating examples (harvest67.py). Arguments are passed to harvest67.py.
# CPU job: the data phase (streams the Pile) and the merge. Needs HARVEST_SNAP.
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J harvest67
#SBATCH -o logs/harvest67_cpu_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 RAYON_NUM_THREADS=16 WANDB_MODE=disabled
snap="${HARVEST_SNAP:?set HARVEST_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/harvest67.py" "$@"
echo HARVEST67_JOB_DONE
