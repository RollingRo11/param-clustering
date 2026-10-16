#!/bin/bash
# The VPD paper's princess graph scored and re-edged on the published checkpoint (paper_graph67.py). Arguments are passed through. Needs GRAPH_SNAP.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J papergraph67
#SBATCH -o logs/papergraph67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
snap="${GRAPH_SNAP:?set GRAPH_SNAP}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$snap/paper_graph67.py" "$@"
echo PAPERGRAPH67_JOB_DONE
