#!/bin/bash
# Circuits at emoticon starters and the VPD paper's emoticon edit for one decomposition (emoticon67.py).
# Arguments: METHOD CHECKPOINT OUTPUT_JSON. Needs EMOTICON_SNAP (with emoticon_tokens.json).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J emoticon67
#SBATCH -o logs/emoticon67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${EMOTICON_SNAP:?set EMOTICON_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/emoticon67.py" --method "$1" --checkpoint "$2" --output "$3"
echo EMOTICON_JOB_DONE
