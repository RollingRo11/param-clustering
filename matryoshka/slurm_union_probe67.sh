#!/bin/bash
# What the never-needed pieces do (union_probe67.py, after aggregation67.py) for one decomposition.
# Arguments: METHOD CHECKPOINT OUTPUT_JSON [extra union_probe67.py arguments, e.g. --recipients 32 --donors 32]. Needs UNION_SNAP.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J unionprobe67
#SBATCH -o logs/unionprobe67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${UNION_SNAP:?set UNION_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/union_probe67.py" --method "$1" --checkpoint "$2" --output "$3" "${@:4}"
echo UNION_PROBE_JOB_DONE
