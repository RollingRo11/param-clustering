#!/bin/bash
# 13B-token Matryoshka run (matryoshka13b.py) on 4 H100s on VPD's published-run token stream.
# Finished run: nothing to do. No checkpoint yet: the unit tests and a 20-step 4-GPU smoke (separate folder) first,
# then the run. A checkpoint: resume. A pause before the time limit (OUT/paused.json from this job): the next job in the chain
# (submitted with --dependency=afterany) resumes it. No W&B here: wandb_relay.py forwards metrics.jsonl from a login node.
#SBATCH -A hammerhai-parallax
#SBATCH -p kisski-h100
#SBATCH --gres=gpu:H100:4
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH --signal=USR1@900
#SBATCH -J matryoshka-13b
#SBATCH -o /mnt/vast-kisski/projects/hammerhai-parallax/matryoshka13b/logs/matryoshka13b_%j.log
set -euo pipefail
M=/mnt/vast-kisski/projects/hammerhai-parallax/matryoshka13b
VENV=/mnt/vast-kisski/projects/hammerhai-parallax/parauto13b/code/vpd/.venv
END=$(( $(date +%s) + 172800 - 1500 ))                    # the job limit minus 25 minutes
export PAIR_ROOT=$M HF_HOME=$M/hf OMP_NUM_THREADS=8 WANDB_MODE=disabled
export MATRYOSHKA_FORWARD=hard MATRYOSHKA_SENSOR=fisher MATRYOSHKA_LP_COEFF=1 MATRYOSHKA_BUDGET_STRATIFIED=1 \
       MATRYOSHKA_BUDGET_CONTEXTS=1 MATRYOSHKA_BUDGET_POWER=1.5 MATRYOSHKA_MASS_COEFF=10
cd $M/code/compare
sha256sum --check --quiet SHA256SUMS
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
OUT=$M/output/matryoshka13b
if [ -f $OUT/result.json ]; then echo "RUN_ALREADY_FINISHED"; exit 0; fi
run="$VENV/bin/torchrun --standalone --nproc_per_node=4"
if [ ! -f $OUT/latest_checkpoint.json ]; then
  $VENV/bin/python -u test_matryoshka13b.py
  rm -rf $M/output/matryoshka13b_smoke
  SMOKE_STEPS=20 RUN_DEADLINE=$END $run matryoshka13b.py
  echo SMOKE_PASSED
fi
set +e
RUN_DEADLINE=$END $run matryoshka13b.py
status=$?
set -e
if [ -f $OUT/result.json ]; then echo MATRYOSHKA_13B_DONE; exit 0; fi
# torchrun turns the worker's exit 3 into 1: the pause is recognized by its marker, written by this job.
if $VENV/bin/python -c "import json, sys; sys.exit(json.load(open('$OUT/paused.json'))['job'] != '${SLURM_JOB_ID}')" 2>/dev/null; then
  echo PAUSED_FOR_CONTINUATION; exit 0
fi
echo "RUN_FAILED status $status"; exit 1
