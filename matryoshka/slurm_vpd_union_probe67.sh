#!/bin/bash
# Why VPD does not recover with every needed component on (vpd_union_probe67.py, after aggregation67.py).
# Arguments: CHECKPOINT OUTPUT_JSON [extra vpd_union_probe67.py arguments, e.g. --recipients 32 --donors 32]. Needs UNION_SNAP.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J vpdunion67
#SBATCH -o logs/vpdunion67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${UNION_SNAP:?set UNION_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/vpd_union_probe67.py" --checkpoint "$1" --output "$2" "${@:3}"
echo VPD_UNION_PROBE_JOB_DONE
