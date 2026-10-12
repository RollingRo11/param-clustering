#!/bin/bash
# Shared setup for compute allocations. Never print env.sh or its secrets.
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
parauto_source="${PARAUTO_SOURCE:-$PWD/code/par-auto-current}"
parauto_assets="$COFAC_DATA/parauto/spd_assets_10241587"
parauto_target="$COFAC_DATA/fission/paper_target_10219891/target/target.pt"
export PYTHONPATH="$parauto_assets/deps:$PWD/code/spd-paper-reference:$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_PROJECT=param-autoenc
export WANDB_MODE="${WANDB_MODE:-online}"
(cd "$parauto_source" && sha256sum -c SHA256SUMS)
