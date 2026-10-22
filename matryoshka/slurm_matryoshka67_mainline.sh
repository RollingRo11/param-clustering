#!/bin/bash
# Matryoshka 67M mainline recipe (21 Oct 2026) from the git clone on Explorer (MATRYOSHKA_REPO, default
# /projects/RohanKathuria/param-clustering/repo; update it with git pull). Only committed code runs: the job stops if
# the clone has local changes, and a resumed run stops if the clone's commit differs from the one the run started on.
# Usage, from /projects/RohanKathuria/param-clustering:
#   sbatch repo/matryoshka/slurm_matryoshka67_mainline.sh smoke   tests, then 20 steps into .../smoke (not on W&B)
#   sbatch repo/matryoshka/slurm_matryoshka67_mainline.sh train   resumable; exit 75 at the time budget resubmits
#                                                                  itself (at most 30 jobs); no eval is submitted
# Env (optional): MATRYOSHKA_STEPS (default 30518 = 1B tokens; the lr cosine, p schedule and attacker warm-up are
# stretched to it), MATRYOSHKA_RUN (run folder name, default mainline), MATRYOSHKA_TIME_BUDGET (seconds of training
# per job, default 6300 for the 2 h limit), MATRYOSHKA_SBATCH_EXTRA (extra sbatch options for continuations).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@300
#SBATCH -J matryoshka67-mainline
#SBATCH -o logs/matryoshka67_mainline_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source ./env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
repo="${MATRYOSHKA_REPO:-$PWD/repo}"
code="$repo/matryoshka"
commit=$(git -C "$repo" rev-parse HEAD)
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then echo "UNCOMMITTED CHANGES in $repo"; exit 1; fi
echo "COMMIT $commit"

# The mainline recipe (main.md; knobs in newmethod.md).
export MATRYOSHKA_FORWARD=hard MATRYOSHKA_SENSOR=fisher MATRYOSHKA_K_MIN=32 \
  MATRYOSHKA_BUDGET_POWER=1.5 MATRYOSHKA_BUDGET_CONTEXTS=1 MATRYOSHKA_BUDGET_STRATIFIED=1 \
  MATRYOSHKA_LP_COEFF=1 MATRYOSHKA_MASS_COEFF=10 MATRYOSHKA_FREQ_COEFF=0.5 MATRYOSHKA_TAIL_COEFF=1 \
  MATRYOSHKA_ADV_FRACTION=0.9 MATRYOSHKA_PARTICLES=1 \
  MATRYOSHKA_ADV_LR=0.02 MATRYOSHKA_ADV_BETA1=0.01 MATRYOSHKA_ADV_BETA2=0.99 \
  MATRYOSHKA_ROUTING=uniform_k_subset MATRYOSHKA_PROJECTION=fp64
steps="${MATRYOSHKA_STEPS:-30518}"
export MATRYOSHKA_STEPS="$steps" MATRYOSHKA_SCHEDULE_STEPS="$steps" MATRYOSHKA_MONITOR_EVERY="${MATRYOSHKA_MONITOR_EVERY:-250}"
out="$PAIR_ROOT/output/matryoshka67_mainline"
mkdir -p "$out"
mode="${1:-train}"
cd "$PAIR_ROOT/code/vpd"          # its venv has param_decomp (the target model class)
py=.venv/bin/python

if [ "$mode" = smoke ]; then
  $py -u "$code/test_matryoshka67.py"
  rm -rf "$out/smoke"
  MATRYOSHKA_OUT="$out/smoke" MATRYOSHKA_STEPS=20 MATRYOSHKA_MONITOR_EVERY=10 WANDB_MODE=disabled \
    MATRYOSHKA_DEADLINE=$(( $(date +%s) + 5400 )) $py -u "$code/matryoshka67.py"
  rm -f "$out"/smoke/checkpoints/*/checkpoint.pt      # the 17 GB attacker state; the model.pth files stay
  echo "MATRYOSHKA_SMOKE_PASSED $commit"
  exit 0
fi

test "$mode" = train
run="$out/${MATRYOSHKA_RUN:-mainline}"
mkdir -p "$run"
if [ -f "$run/commits.txt" ]; then
  first=$(head -1 "$run/commits.txt" | cut -d' ' -f2)
  if [ "$first" != "$commit" ]; then echo "RUN STARTED ON $first, CLONE IS AT $commit"; exit 1; fi
fi
echo "${SLURM_JOB_ID} $commit $(date -Is)" >> "$run/commits.txt"
export MATRYOSHKA_OUT="$run" MATRYOSHKA_DEADLINE=$(( $(date +%s) + ${MATRYOSHKA_TIME_BUDGET:-6300} ))
set +e
$py -u "$code/matryoshka67.py"
status=$?
set -e
cd /projects/RohanKathuria/param-clustering
if [ "$status" -eq 75 ]; then
  chunk="${MATRYOSHKA_CHUNK:-0}"
  if [ "$chunk" -ge 30 ]; then echo CONTINUATION_LIMIT_CHECKPOINT_SAVED; exit 75; fi
  next=$(sbatch --parsable ${MATRYOSHKA_SBATCH_EXTRA:-} --dependency="afterok:$SLURM_JOB_ID" \
    --export="ALL,MATRYOSHKA_CHUNK=$((chunk+1))" "$code/slurm_matryoshka67_mainline.sh" train)
  printf '%s %s %s\n' "$SLURM_JOB_ID" "$next" "$chunk" >> "$run/continuations.txt"
  echo "CONTINUATION $next"
  exit 0
fi
exit "$status"
