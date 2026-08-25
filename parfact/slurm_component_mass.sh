#!/bin/bash
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH -o logs/component_mass.log
# Per-component weight mass for the two C=600 v2 runs.
#
# /cofact is a Beam DurableDisk and is NOT visible here, so step 1 stages the
# two factorization.pt files from the R2 mirror of that disk onto /projects.
set -euo pipefail

cd /projects/RohanKathuria/param-clustering
source env.sh

PY=code/.venv/bin/python
STAGE=data/component_mass
SS=B_layer_K1200_C600_v2_idiv_usimplex_ssimplex
US=B_layer_K1200_C600_v2_idiv_usimplex

$PY -u code/r2_pull_fact.py --dest "$STAGE" "$SS" "$US"

$PY -u code/plot_component_mass.py \
    --usimplex "$STAGE/$SS" \
    --baseline "$STAGE/$US" \
    --usimplex_label "U+S simplex (v2 idiv, u+s)" \
    --baseline_label "U simplex only (v2 idiv, u)" \
    --title "Component mass: U+S simplex vs U simplex only" \
    --out out/component_mass

echo "--- results ---"
ls -la out/component_mass
