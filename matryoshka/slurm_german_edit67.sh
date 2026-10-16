#!/bin/bash
# German removal by editing (german_edit67.py) on the 67M target. Arguments are passed to german_edit67.py.
# Needs GERMAN_SNAP (the code snapshot). GPU phases: rank, edit; the data phase also runs here (it needs the network).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J german67
#SBATCH -o logs/german67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${GERMAN_SNAP:?set GERMAN_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/german_edit67.py" "$@"
echo GERMAN67_JOB_DONE
