#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -J parauto-ci-census
#SBATCH -o logs/parauto_ci_census_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/faithfulness"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/ci_census.py" --method vpd --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/census_vpd_published.json"
.venv/bin/python -u "$compare/ci_census.py" --method vpd --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" --output "$out/census_vpd_paired.json"
.venv/bin/python -u "$compare/ci_census.py" --method parauto --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/census_parauto_1b.json"
echo CI_CENSUS_ALL_DONE
