#!/bin/bash
# Self-repair correlation kappa on held-out val rows: par-auto GIM 5k vs published VPD.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J parauto-repair-kappa
#SBATCH -o logs/parauto_repair_kappa_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/repair_kappa_20260916"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/repair_kappa_probe.py" --self-test
run() { .venv/bin/python -u "$compare/repair_kappa_probe.py" "$@"; }
run --name gim_5000      --sequences 64 --fisher-sequences 8 --output "$out/gim_5000.json"
run --name vpd_published --sequences 64 --fisher-sequences 8 --output "$out/vpd_published.json"
echo REPAIR_KAPPA_ALL_DONE
