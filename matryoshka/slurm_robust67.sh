#!/bin/bash
# Matryoshka par-auto robustness variants (robust67.py), 5,000 steps at the 5k mass-10 settings: hard forward, Fisher
# score, Lp 1, stratified k^-1.5 budgets one per context, mass penalty 10. Logs to W&B project
# rohan-kathuria-neu/matryoshka; smokes do not log. Arguments: MODE VARIANT [COEFF=1]
#   MODE smoke: 20 steps into output/matryoshka67_20260923/smoke_robust_VARIANT (not logged to W&B)
#   MODE train: resumable into output/matryoshka67_20260923/robust_VARIANT_cCOEFF; exit 75 at the time budget
#               resubmits this script; completion submits the KLKeep eval, which submits VPD's adversarial eval.
#   VARIANT: energy | union | holes | attack | prefix (see robust67.py)
# Code snapshot: MATRYOSHKA_SNAP (default robust67_20261015), carried into continuations and the eval jobs.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@180
#SBATCH -J robust67
#SBATCH -o logs/robust67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mode="${1:?MODE}" variant="${2:?VARIANT}" coeff="${3:-1}"
export MATRYOSHKA_FORWARD=hard MATRYOSHKA_SENSOR=fisher MATRYOSHKA_LP_COEFF=1 MATRYOSHKA_BUDGET_STRATIFIED=1 \
  MATRYOSHKA_BUDGET_CONTEXTS=1 MATRYOSHKA_BUDGET_POWER=1.5 MATRYOSHKA_MASS_COEFF=10 \
  MATRYOSHKA_ROBUST="$variant" MATRYOSHKA_ROBUST_COEFF="$coeff"
export MATRYOSHKA_SNAP="${MATRYOSHKA_SNAP:-$PAIR_ROOT/code/robust67_20261015}"
snap="$MATRYOSHKA_SNAP"
out="$PAIR_ROOT/output/matryoshka67_20260923"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
if [ "$mode" = smoke ]; then
  rm -rf "$out/smoke_robust_$variant"
  MATRYOSHKA_OUT="$out/smoke_robust_$variant" MATRYOSHKA_STEPS=20 MATRYOSHKA_MONITOR_EVERY=10 WANDB_MODE=disabled \
    MATRYOSHKA_DEADLINE=$(( $(date +%s) + 6300 )) .venv/bin/python -u "$snap/robust67.py"
  echo "ROBUST_SMOKE_PASSED $variant"
  exit 0
fi
test "$mode" = train
run="$out/robust_${variant}_c$coeff"
export MATRYOSHKA_OUT="$run"
export MATRYOSHKA_DEADLINE=$(( $(date +%s) + ${MATRYOSHKA_TIME_BUDGET:-6300} ))
set +e
.venv/bin/python -u "$snap/robust67.py"
status=$?
set -e
cd /projects/RohanKathuria/param-clustering
if [ "$status" -eq 75 ]; then
  chunk="${MATRYOSHKA_CHUNK:-0}"
  if [ "$chunk" -ge 8 ]; then echo CONTINUATION_LIMIT_CHECKPOINT_SAVED; exit 75; fi
  next=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" --export="ALL,MATRYOSHKA_CHUNK=$((chunk+1))" \
    "$snap/slurm_robust67.sh" train "$variant" "$coeff")
  printf '%s %s %s\n' "$SLURM_JOB_ID" "$next" "$chunk" >> "$run/continuations.txt"
  echo "CONTINUATION $next"
  exit 0
fi
if [ "$status" -ne 0 ]; then exit "$status"; fi
next=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" "$snap/slurm_matryoshka67_eval.sh" "$run" fisher)
printf '%s\n' "$next" > "$run/eval_submission.txt"
echo "EVAL $next"
