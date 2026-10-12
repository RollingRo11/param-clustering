#!/bin/bash
# Anatomy of VPD's adversarial eval on one decomposition (adversary_anatomy67.py): real-text cancellation, what the
# dropped pieces carry, and which pieces and layers the 20-step attack uses. Arguments: METHOD CHECKPOINT OUTPUT_JSON. Needs ANATOMY_SNAP.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J anatomy67
#SBATCH -o logs/anatomy67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${ANATOMY_SNAP:?set ANATOMY_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/adversary_anatomy67.py" --method "$1" --checkpoint "$2" --output "$3"
echo ADVERSARY_ANATOMY_JOB_DONE
