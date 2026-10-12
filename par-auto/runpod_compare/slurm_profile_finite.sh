#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:50:00
#SBATCH -J parauto-finite-prof
#SBATCH -o logs/parauto_finite_prof_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_finite_ablation.py"
.venv/bin/python -u "$compare/profile_finite.py"
.venv/bin/python -u "$compare/smoke_finite.py"
# Resume check in a scratch folder: stop on a short budget, then --resume and stop again.
export FINITE_OUT="$PAIR_ROOT/output/smoke/finite_resume_$SLURM_JOB_ID" WANDB_MODE=disabled
GIM_DEADLINE=$(( $(date +%s) + 150 )) .venv/bin/python -u "$compare/train_finite_5000.py" || echo PART1_STOPPED_AS_EXPECTED
GIM_DEADLINE=$(( $(date +%s) + 120 )) .venv/bin/python -u "$compare/train_finite_5000.py" --resume || echo PART2_STOPPED_AS_EXPECTED
cat "$FINITE_OUT"/result_*.json "$FINITE_OUT/latest_checkpoint.json"
echo RESUME_SMOKE_DONE
