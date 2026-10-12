#!/bin/bash
# Layer-local sufficiency of sparse supports: published VPD (its own CI support) vs GIM 5k at matched counts.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -J parauto-local-suff
#SBATCH -o logs/parauto_local_suff_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/local_sufficiency_20260915"
cd "$PAIR_ROOT/code/vpd"
run() { .venv/bin/python -u "$compare/local_sufficiency_probe.py" "$@"; }
run --name vpd_published --support native                --output "$out/vpd_published_native.json"
run --name vpd_published --support count --count 193     --output "$out/vpd_published_193.json"
run --name gim_5000      --support count --count 193     --output "$out/gim_5000_193.json"
run --name gim_5000      --support count --count 512     --output "$out/gim_5000_512.json"
run --name vpd_published --support count --count 512     --output "$out/vpd_published_512.json"
echo LOCAL_SUFFICIENCY_DONE
