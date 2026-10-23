#!/bin/bash
# End-of-run evals of a SimpleStories run: KLKeep (128 val rows) ranked three ways (per-token Fisher K=8, as for the
# 20 Oct SS runs; the cached text-averaged Fisher, static_fisher.py; the write size |r.h| ||u||), the global adversarial
# probe (128 sequences, keeps 0.5% / 1% / 10%) ranked by Fisher and, for a run trained with another sensor, by that one;
# then log_evals_ss.py attaches them to the run's W&B training run. Runs trained with another sensor score the static
# ranking with the target's own metric, cached at $MATRYOSHKA_STATIC_CACHE.
# Usage: matryoshka/eval_ss_run.sh RUN_DIR [STEP]   (STEP: default the run's last completed update)
set -euo pipefail
run=$1
code="$(cd "$(dirname "$0")" && pwd)"
export MATRYOSHKA_TARGET=simplestories SS_DATA="${SS_DATA:-/root/data/ss}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="${VPD_CODE:-/root/vpd}${PYTHONPATH:+:$PYTHONPATH}" MATRYOSHKA_WANDB_DIR=/root/wandb_eval
export MATRYOSHKA_STATIC_CACHE="${MATRYOSHKA_STATIC_CACHE:-$SS_DATA/static_fisher.pt}"
py="${PY:-/root/venvs/ss/bin/python}"
step="${2:-$(cd "$code" && $py -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$run/latest_checkpoint.json")}"
ckpt="$run/checkpoints/$(printf %06d "$step")/model.pth"
sensor=$($py -c "import json,sys; print(json.load(open(sys.argv[1]))['sensor'])" "$run/plan.json")
name=$(basename "$run")
mkdir -p "$MATRYOSHKA_WANDB_DIR"
cd "$code"
klkeep() {     # sensor, folder
  [ -f "$run/$2/klkeep.json" ] || $py -u evaluate_checkpoint.py parauto --expected-step "$step" --sensor "$1" --fisher-k 8 \
    --checkpoint "$ckpt" --output "$run/$2" --run-tag "ss-$name-$step-$1-klkeep"
}
probe() {      # sensor, output file
  [ -f "$run/$2" ] || WANDB_MODE=disabled $py -u adversarial_probe.py --checkpoint "$ckpt" --output "$run/$2" --sensor "$1" \
    --scope global --sequences 128 --microbatch 16 --keeps 0.005,0.01,0.1
}
# KLKeep under all three rankings: per-token Fisher K=8, the cached text-averaged Fisher, the plain write size.
klkeep fisher eval_fisher_k8
klkeep static eval_static
klkeep write eval_write
probe fisher adversarial_global_128.json
[ "$sensor" = fisher ] || probe "$sensor" "adversarial_global_128_$sensor.json"
$py -u log_evals_ss.py "$run"
echo "EVALS_DONE $name $step"
