#!/bin/bash
# Adversarial masking probe (adversarial_probe.py) from a code snapshot, e.g. with --adversary hard / --init off.
# Arguments: CHECKPOINT OUTPUT_JSON [extra adversarial_probe.py arguments]. Needs PROBE_SNAP.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J adversarial67
#SBATCH -o logs/adversarial67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${PROBE_SNAP:?set PROBE_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/adversarial_probe.py" --checkpoint "$1" --output "$2" "${@:3}"
echo ADVERSARIAL_PROBE_JOB_DONE
