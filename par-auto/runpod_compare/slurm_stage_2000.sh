#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J paired-2k-stage
#SBATCH -o logs/paired_2k_stage_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
code/.venv/bin/python -u "$compare/stage_milestone.py" --setup
cd "$PAIR_ROOT/code/vpd"
~/.local/bin/uv sync --locked --no-dev
~/.local/bin/uv pip install --python .venv/bin/python boto3
.venv/bin/python "$compare/evaluate_checkpoint.py" --help
.venv/bin/python -u "$compare/stage_milestone.py" --step 2000 --hours 3
