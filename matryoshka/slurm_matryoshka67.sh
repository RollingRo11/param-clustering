#!/bin/bash
# Matryoshka par-auto: sigmoid top-k over the read x write score, masked-forward KL, 5,000 steps. Training runs log to
# W&B project rohan-kathuria-neu/matryoshka; smokes, pilots and evals do not log to W&B.
# Arguments: MODE [FORWARD=soft|hard] [SENSOR=fisher|gim] [LP_COEFF=0]. Run directories: main (soft, fisher),
# main_hard (hard, fisher), a _gim suffix for the GIM score (e.g. main_hard_gim) and _lpC for an Lp term of weight C
# (e.g. main_hard_gim_lp0.1); pilots and smokes follow the same tags.
# Environment (optional, carried into continuations and the eval job): MATRYOSHKA_SNAP (code snapshot directory,
# default matryoshka67_20260923), MATRYOSHKA_BUDGET_STRATIFIED=1 (suffix _strat), MATRYOSHKA_BUDGET_CONTEXTS=N != 4
# (suffix _perN), MATRYOSHKA_BUDGET_POWER=P != 1 (suffix _powP), MATRYOSHKA_ADV_COEFF=C != 0 (VPD's persistent PGD loss
# with weight C, suffix _advC), MATRYOSHKA_ADV_GATE=soft (the adversary's keep level is the sigmoid top-k alpha
# instead of the hard keep set, suffix _advgatesoft), MATRYOSHKA_MASS_COEFF=C != 0 (piece-mass penalty with weight C, suffix _massC),
# MATRYOSHKA_FREQ_BETA=B != 0 (VPD's frequency term on the Lp term, suffix _freqB),
# MATRYOSHKA_CERT_GAMMA=G != 0 (signed tail certificate penalty with weight G and MATRYOSHKA_CERT_TAU, default 0.25,
# suffix _certG_tauT), MATRYOSHKA_INIT_TAG=T with MATRYOSHKA_INIT=<checkpoint.pt> (start from another run's full
# checkpoint, its step and lr schedule included; suffix _from_T),
# MATRYOSHKA_HARD_PROBE_SNAP (the eval job then also submits the hard adversarial probes from that snapshot).
#   smoke: tests + 20 steps into .../smoke_TAG (not logged to W&B)
#   pilot: tests + the first 300 steps into .../pilot_TAG (not logged to W&B)
#   train: resumable; exit 75 at the time budget resubmits this script, completion submits the KLKeep eval.
#     Optional for one long job instead of a chain: MATRYOSHKA_TEST_FIRST=1 (tests before a fresh start, in place of a
#     smoke job), MATRYOSHKA_TIME_BUDGET (seconds of training before pausing; default 6300 for this 2-hour limit),
#     MATRYOSHKA_SBATCH_EXTRA (extra sbatch options for continuations, e.g. "-p gpu -t 08:00:00"),
#     MATRYOSHKA_EVAL_INLINE=1 (run the eval, and its adversarial-masking job, inside this job).
#     MATRYOSHKA_EVAL_STEP (the step the eval loads; default 5000, set it to MATRYOSHKA_STEPS for a fine-tune).
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@180
#SBATCH -J matryoshka67
#SBATCH -o logs/matryoshka67_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mode="${1:-train}"
export MATRYOSHKA_FORWARD="${2:-soft}" MATRYOSHKA_SENSOR="${3:-fisher}" MATRYOSHKA_LP_COEFF="${4:-0}"
suffix=""
if [ "$MATRYOSHKA_SENSOR" != fisher ]; then suffix="${suffix}_$MATRYOSHKA_SENSOR"; fi
if [ "$MATRYOSHKA_LP_COEFF" != 0 ]; then suffix="${suffix}_lp$MATRYOSHKA_LP_COEFF"; fi
if [ "${MATRYOSHKA_BUDGET_STRATIFIED:-0}" = 1 ]; then suffix="${suffix}_strat"; fi
if [ "${MATRYOSHKA_BUDGET_CONTEXTS:-4}" != 4 ]; then suffix="${suffix}_per$MATRYOSHKA_BUDGET_CONTEXTS"; fi
if [ "${MATRYOSHKA_BUDGET_POWER:-1}" != 1 ]; then suffix="${suffix}_pow$MATRYOSHKA_BUDGET_POWER"; fi
if [ "${MATRYOSHKA_ADV_COEFF:-0}" != 0 ]; then suffix="${suffix}_adv$MATRYOSHKA_ADV_COEFF"; fi
if [ "${MATRYOSHKA_ADV_GATE:-hard}" != hard ]; then suffix="${suffix}_advgate$MATRYOSHKA_ADV_GATE"; fi
if [ "${MATRYOSHKA_FREQ_BETA:-0}" != 0 ]; then suffix="${suffix}_freq$MATRYOSHKA_FREQ_BETA"; fi
if [ "${MATRYOSHKA_MASS_COEFF:-0}" != 0 ]; then suffix="${suffix}_mass$MATRYOSHKA_MASS_COEFF"; fi
if [ "${MATRYOSHKA_CERT_GAMMA:-0}" != 0 ]; then suffix="${suffix}_cert${MATRYOSHKA_CERT_GAMMA}_tau${MATRYOSHKA_CERT_TAU:-0.25}"; fi
if [ -n "${MATRYOSHKA_INIT:-}" ]; then suffix="${suffix}_from_${MATRYOSHKA_INIT_TAG:?MATRYOSHKA_INIT needs MATRYOSHKA_INIT_TAG}"; fi
tag="$MATRYOSHKA_FORWARD$suffix"
snap="${MATRYOSHKA_SNAP:-$PAIR_ROOT/code/matryoshka67_20260923}"
out="$PAIR_ROOT/output/matryoshka67_20260923"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
if [ "$mode" = smoke ] || [ "$mode" = pilot ]; then
  steps=20; every=10
  if [ "$mode" = pilot ]; then steps=300; every=50; fi
  .venv/bin/python -u "$snap/test_matryoshka67.py"
  rm -rf "$out/${mode}_$tag"
  MATRYOSHKA_OUT="$out/${mode}_$tag" MATRYOSHKA_STEPS=$steps MATRYOSHKA_MONITOR_EVERY=$every WANDB_MODE=disabled \
    MATRYOSHKA_DEADLINE=$(( $(date +%s) + 6300 )) .venv/bin/python -u "$snap/matryoshka67.py"
  echo "MATRYOSHKA_${mode^^}_PASSED"
  exit 0
fi
test "$mode" = train
run="$out/main"
if [ "$MATRYOSHKA_FORWARD" = hard ]; then run="$out/main_hard"; fi
run="$run$suffix"
export MATRYOSHKA_OUT="$run"
if [ "${MATRYOSHKA_TEST_FIRST:-0}" = 1 ] && [ ! -f "$run/latest_checkpoint.json" ]; then .venv/bin/python -u "$snap/test_matryoshka67.py"; fi
export MATRYOSHKA_DEADLINE=$(( $(date +%s) + ${MATRYOSHKA_TIME_BUDGET:-6300} ))
set +e
.venv/bin/python -u "$snap/matryoshka67.py"
status=$?
set -e
cd /projects/RohanKathuria/param-clustering
if [ "$status" -eq 75 ]; then
  chunk="${MATRYOSHKA_CHUNK:-0}"
  if [ "$chunk" -ge 8 ]; then echo CONTINUATION_LIMIT_CHECKPOINT_SAVED; exit 75; fi
  next=$(sbatch --parsable ${MATRYOSHKA_SBATCH_EXTRA:-} --dependency="afterok:$SLURM_JOB_ID" --export="ALL,MATRYOSHKA_CHUNK=$((chunk+1))" \
    "$snap/slurm_matryoshka67.sh" train "$MATRYOSHKA_FORWARD" "$MATRYOSHKA_SENSOR" "$MATRYOSHKA_LP_COEFF")
  printf '%s %s %s\n' "$SLURM_JOB_ID" "$next" "$chunk" >> "$run/continuations.txt"
  echo "CONTINUATION $next"
  exit 0
fi
if [ "$status" -ne 0 ]; then exit "$status"; fi
if [ "${MATRYOSHKA_EVAL_INLINE:-0}" = 1 ]; then
  echo "EVAL inline"
  bash "$snap/slurm_matryoshka67_eval.sh" "$run" "$MATRYOSHKA_SENSOR"
  exit 0
fi
next=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" "$snap/slurm_matryoshka67_eval.sh" "$run" "$MATRYOSHKA_SENSOR")
printf '%s\n' "$next" > "$run/eval_submission.txt"
echo "EVAL $next"
