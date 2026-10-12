#!/bin/bash
# Explanation aggregation (aggregation67.py, after the VPD audit on LessWrong) for one decomposition.
# Arguments: METHOD CHECKPOINT OUTPUT_JSON [extra aggregation67.py arguments, e.g. --smoke]. Needs AGGREGATION_SNAP.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J aggregation67
#SBATCH -o logs/aggregation67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${AGGREGATION_SNAP:?set AGGREGATION_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/aggregation67.py" --method "$1" --checkpoint "$2" --output "$3" "${@:4}"
echo AGGREGATION_JOB_DONE
