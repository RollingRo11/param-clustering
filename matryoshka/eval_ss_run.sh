#!/bin/bash
# End-of-run evals of a SimpleStories run, as for the 20 Oct SS runs: KLKeep (128 val rows, Fisher K=8 ranking), the
# global adversarial probe (128 sequences, keeps 0.5% / 1% / 10%), and, for a run ranked by another sensor, both again
# with that run's own ranking; then log_evals_ss.py attaches them to the run's W&B training run.
# Usage: matryoshka/eval_ss_run.sh RUN_DIR [STEP]   (STEP: default the run's last completed update)
set -euo pipefail
run=$1
code="$(cd "$(dirname "$0")" && pwd)"
export MATRYOSHKA_TARGET=simplestories SS_DATA="${SS_DATA:-/root/data/ss}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="${VPD_CODE:-/root/vpd}${PYTHONPATH:+:$PYTHONPATH}" MATRYOSHKA_WANDB_DIR=/root/wandb_eval
py="${PY:-/root/venvs/ss/bin/python}"
step="${2:-$(cd "$code" && $py -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$run/latest_checkpoint.json")}"
ckpt="$run/checkpoints/$(printf %06d "$step")/model.pth"
sensor=$($py -c "import json,sys; print(json.load(open(sys.argv[1]))['sensor'])" "$run/plan.json")
name=$(basename "$run")
mkdir -p "$MATRYOSHKA_WANDB_DIR"
cd "$code"
evaluate() {   # sensor, KLKeep folder, probe file
  [ -f "$run/$2/klkeep.json" ] || $py -u evaluate_checkpoint.py parauto --expected-step "$step" --sensor "$1" --fisher-k 8 \
    --checkpoint "$ckpt" --output "$run/$2" --run-tag "ss-$name-$step-$1-klkeep"
  [ -f "$run/$3" ] || WANDB_MODE=disabled $py -u adversarial_probe.py --checkpoint "$ckpt" --output "$run/$3" --sensor "$1" \
    --scope global --sequences 128 --microbatch 16 --keeps 0.005,0.01,0.1
}
evaluate fisher eval_fisher_k8 adversarial_global_128.json
[ "$sensor" = fisher ] || evaluate "$sensor" "eval_$sensor" "adversarial_global_128_$sensor.json"
$py -u log_evals_ss.py "$run"
echo "EVALS_DONE $name $step"
