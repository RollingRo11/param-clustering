#!/bin/bash
# Matryoshka mainline recipe (21 Oct 2026; the same knobs as slurm_matryoshka67_mainline.sh) on the 2-layer SimpleStories
# target (gf6rbga0, 7,104 pieces), on a single-GPU box from this git clone. Only committed code runs: the script stops if
# the clone has local changes, and a resumed run stops if the clone's commit differs from the one the run started on.
# Usage:
#   matryoshka/run_ss_mainline.sh smoke   tests on the SS target, then 20 steps into $SS_OUT/smoke (not on W&B)
#   matryoshka/run_ss_mainline.sh train   resumable (rerun the same command to continue from the newest full checkpoint)
# Env (optional): MATRYOSHKA_STEPS (default: one epoch of the train split, 18,522 updates = 607M tokens; the lr cosine,
# p schedule and attacker warm-up are stretched to it unless MATRYOSHKA_SCHEDULE_STEPS sets another horizon; the earlier
# 5,000-step SS runs used 8,700), MATRYOSHKA_K_MIN (smallest budget, default 32), MATRYOSHKA_SENSOR (default fisher;
# write = rank by |r.h| ||u||, one forward; static = |r.h| sqrt(u^T M u)), MATRYOSHKA_BUDGET_POWER (budget density
# k^-power, default 1.5), MATRYOSHKA_MONITOR_EVERY (default 250),
# Supersparse budgets: MATRYOSHKA_K_MIN=8 MATRYOSHKA_BUDGET_POWER=2 (median budget 16, 89% of draws at k <= 71, about
# VPD's 47 active pieces per token; 1% at k >= 710, so the dense end is still trained).
#
# MATRYOSHKA_RUN (run folder name, default mainline), SS_DATA (target + token rows, default /root/data/ss),
# SS_OUT (run folders, default /root/runs/matryoshka_ss_mainline), PY (python with torch, wandb and
# param_decomp's model classes; default /root/venvs/ss/bin/python), VPD_CODE (param_decomp checkout for PYTHONPATH).
# W&B: project matryoshka, group matryoshka_ss_mainline; log in once with `wandb login` (the key stays in ~/.netrc).
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
code="$repo/matryoshka"
commit=$(git -C "$repo" rev-parse HEAD)
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then echo "UNCOMMITTED CHANGES in $repo"; exit 1; fi
echo "COMMIT $commit"

export MATRYOSHKA_TARGET=simplestories SS_DATA="${SS_DATA:-/root/data/ss}"
export PYTHONPATH="${VPD_CODE:-/root/vpd}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
py="${PY:-/root/venvs/ss/bin/python}"

# The mainline recipe (main.md; knobs in newmethod.md), as on 67M, except the attacker's lr is constant from the first
# update (MATRYOSHKA_ADV_LR_WARMUP=0), as in the SimpleStories runs it was measured on (particles_ss.py).
export MATRYOSHKA_FORWARD=hard MATRYOSHKA_SENSOR="${MATRYOSHKA_SENSOR:-fisher}" MATRYOSHKA_K_MIN="${MATRYOSHKA_K_MIN:-32}" \
  MATRYOSHKA_BUDGET_POWER="${MATRYOSHKA_BUDGET_POWER:-1.5}" MATRYOSHKA_BUDGET_CONTEXTS=1 MATRYOSHKA_BUDGET_STRATIFIED=1 \
  MATRYOSHKA_LP_COEFF=1 MATRYOSHKA_MASS_COEFF=10 MATRYOSHKA_FREQ_COEFF=0.5 MATRYOSHKA_TAIL_COEFF=1 \
  MATRYOSHKA_ADV_FRACTION=0.9 MATRYOSHKA_PARTICLES=1 \
  MATRYOSHKA_ADV_LR=0.02 MATRYOSHKA_ADV_BETA1=0.01 MATRYOSHKA_ADV_BETA2=0.99 MATRYOSHKA_ADV_LR_WARMUP=0 \
  MATRYOSHKA_ROUTING=uniform_k_subset MATRYOSHKA_PROJECTION=fp64
steps="${MATRYOSHKA_STEPS:-$(cd "$code" && $py -c 'import common_ss; print(common_ss.train_updates())')}"
export MATRYOSHKA_STEPS="$steps" MATRYOSHKA_SCHEDULE_STEPS="${MATRYOSHKA_SCHEDULE_STEPS:-$steps}" MATRYOSHKA_MONITOR_EVERY="${MATRYOSHKA_MONITOR_EVERY:-250}"
out="${SS_OUT:-/root/runs/matryoshka_ss_mainline}"
mkdir -p "$out"
mode="${1:-train}"
cd "$code"

if [ "$mode" = smoke ]; then
  $py -u test_matryoshka67.py
  rm -rf "$out/smoke"
  MATRYOSHKA_OUT="$out/smoke" MATRYOSHKA_STEPS=20 MATRYOSHKA_MONITOR_EVERY=10 WANDB_MODE="${WANDB_MODE:-disabled}" \
    MATRYOSHKA_WANDB_DIR=/root/wandb_smoke MATRYOSHKA_DEADLINE=$(( $(date +%s) + 5400 )) $py -u matryoshka67.py
  rm -f "$out"/smoke/checkpoints/*/checkpoint.pt      # the attacker state; the model.pth files stay
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
echo "$(hostname) $commit $(date -Is)" >> "$run/commits.txt"
# W&B's local files on local disk: its media renames fail on the /workspace network volume.
export MATRYOSHKA_OUT="$run" MATRYOSHKA_WANDB_DIR="/root/wandb/${MATRYOSHKA_RUN:-mainline}" \
  MATRYOSHKA_DEADLINE=$(( $(date +%s) + ${MATRYOSHKA_TIME_BUDGET:-1000000} ))
mkdir -p "$MATRYOSHKA_WANDB_DIR"
exec $py -u matryoshka67.py
