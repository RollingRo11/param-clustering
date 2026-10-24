#!/bin/bash
# THE SIMPLESTORIES RECIPE: "B" (chosen 24 Oct 2026). The defaults below are recipe B; every knob is still overridable.
#   = the 21 Oct mainline (hard top-k, k ~ k^-1.5 on [32, N-1], Lp / mass / hard frequency / tail terms, merged attacker,
#     uniform-k-subset routing) with
#   + static sensor (forward-only |r.h| sqrt(u^T M u), M the cached text-averaged Fisher)    MATRYOSHKA_SENSOR=static
#   + learned per-piece score floors at the pieces' lr                                       MATRYOSHKA_FLOORS=1, _FLOOR_LR_SCALE=1
#   + 30% of contexts kept at their above-floor set                                          MATRYOSHKA_FLOOR_SET_FRACTION=0.3
#   + half the contexts attacked, half plain (was 90/10)                                     MATRYOSHKA_ADV_FRACTION=0.5
#   + 5,000 steps on the 8,700-step schedule, monitor every 100                              MATRYOSHKA_STEPS / _SCHEDULE_STEPS
# Measured (run fs30_adv50_5k): KLKeep 2.31 / 0.50 / 0.41 / 0.30 at k = 21 / 71 / 213 / 710 (VPD 8.22 / 0.43 / 0.38 /
# 0.38); global adversarial probe after 320 steps 12.4 / 2.23 / 0.47 at 0.5 / 1 / 10% kept (VPD 14.3 / 6.08 / 0.48).
# The overnight alternatives (all-masked plain passes, plain loss on attacked contexts, global attack sources: the
# MATRYOSHKA_PLAIN_* / _ATTACK_ROUTING / _ADV_GLOBAL_* knobs) gained KLKeep but lost the probe, so they are off.
#
# Runs on the 2-layer SimpleStories target (gf6rbga0, 7,104 pieces), on a single-GPU box from this git clone. Only committed code runs: the script stops if
# the clone has local changes, and a resumed run stops if the clone's commit differs from the one the run started on.
# Usage:
#   matryoshka/run_ss_mainline.sh smoke   tests on the SS target, then 20 steps into $SS_OUT/smoke (not on W&B)
#   matryoshka/run_ss_mainline.sh train   resumable (rerun the same command to continue from the newest full checkpoint)
# Env (optional): MATRYOSHKA_STEPS (default 5000; one epoch of the train split is 18,522), MATRYOSHKA_SCHEDULE_STEPS
# (horizon of the lr cosine, p schedule and attacker warm-up, default 8700), MATRYOSHKA_K_MIN (smallest budget, default
# 32), MATRYOSHKA_SENSOR (default static; fisher = per-token Fisher K=4, write = |r.h| ||u||), MATRYOSHKA_FLOORS,
# MATRYOSHKA_FLOOR_LR_SCALE, MATRYOSHKA_FLOOR_SET_FRACTION, MATRYOSHKA_ADV_FRACTION (recipe B: 1, 1, 0.3, 0.5),
# MATRYOSHKA_BUDGET_POWER (budget density k^-power, default 1.5), MATRYOSHKA_MONITOR_EVERY (default 100),
# MATRYOSHKA_SCORE_INPUT=masked (static or write sensor): rank each matrix on its masked-forward input (matryoshka67.py).
# Sparser budgets: MATRYOSHKA_K_MIN=8 (power 1.5: median budget 30, 69% of draws at k <= 71, about VPD's 47 active pieces
# per token, 7.5% at k >= 710; power 2: median 16, 89% at k <= 71, 1% at k >= 710).
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

# Recipe B (see the top): the mainline knobs (main.md; newmethod.md), as on 67M except the attacker's lr is constant from
# the first update (MATRYOSHKA_ADV_LR_WARMUP=0, as in particles_ss.py), plus B's static sensor, floors, floor set and 50/50.
export MATRYOSHKA_FORWARD=hard MATRYOSHKA_SENSOR="${MATRYOSHKA_SENSOR:-static}" MATRYOSHKA_K_MIN="${MATRYOSHKA_K_MIN:-32}" \
  MATRYOSHKA_FLOORS="${MATRYOSHKA_FLOORS:-1}" MATRYOSHKA_FLOOR_LR_SCALE="${MATRYOSHKA_FLOOR_LR_SCALE:-1}" \
  MATRYOSHKA_FLOOR_SET_FRACTION="${MATRYOSHKA_FLOOR_SET_FRACTION:-0.3}" \
  MATRYOSHKA_BUDGET_POWER="${MATRYOSHKA_BUDGET_POWER:-1.5}" MATRYOSHKA_BUDGET_CONTEXTS=1 MATRYOSHKA_BUDGET_STRATIFIED=1 \
  MATRYOSHKA_LP_COEFF=1 MATRYOSHKA_MASS_COEFF="${MATRYOSHKA_MASS_COEFF:-10}" MATRYOSHKA_FREQ_COEFF=0.5 MATRYOSHKA_TAIL_COEFF=1 \
  MATRYOSHKA_ADV_FRACTION="${MATRYOSHKA_ADV_FRACTION:-0.5}" MATRYOSHKA_PARTICLES=1 \
  MATRYOSHKA_ADV_LR=0.02 MATRYOSHKA_ADV_BETA1=0.01 MATRYOSHKA_ADV_BETA2=0.99 MATRYOSHKA_ADV_LR_WARMUP=0 \
  MATRYOSHKA_ROUTING=uniform_k_subset MATRYOSHKA_PROJECTION=fp64
export MATRYOSHKA_STEPS="${MATRYOSHKA_STEPS:-5000}" MATRYOSHKA_SCHEDULE_STEPS="${MATRYOSHKA_SCHEDULE_STEPS:-8700}" \
  MATRYOSHKA_MONITOR_EVERY="${MATRYOSHKA_MONITOR_EVERY:-100}"
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
