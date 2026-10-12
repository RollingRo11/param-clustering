#!/bin/bash
# Fetch the latest retained paired-VPD snapshot from R2 (20k if present, else 15k).
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J vpd-late-stage
#SBATCH -o logs/vpd_late_stage_%j.log
set -uo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
cd "$PAIR_ROOT/code/vpd"
for step in 20000 15000; do
  if [ -f "$PAIR_ROOT/checkpoints/step_${step}/inputs_ready.json" ]; then echo "have step_${step}"; break; fi
  .venv/bin/python -u "$PAIR_ROOT/code/compare/stage_retained_snapshot.py" --step $step && { echo "staged step_${step}"; break; }
  rm -rf "$PAIR_ROOT/checkpoints/step_${step}"; echo "no step_${step} in R2"
done
echo STAGE_LATE_DONE
