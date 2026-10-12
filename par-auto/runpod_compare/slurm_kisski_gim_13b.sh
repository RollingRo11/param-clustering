#!/bin/bash
# 13B-token par-auto GIM run on 4 H100s on VPD's published-run token stream.
# Fresh start: 20-step 4-GPU smoke in a separate folder first, then the run. Resume if the run exists.
#SBATCH -A hammerhai-parallax
#SBATCH -p kisski-h100
#SBATCH --gres=gpu:H100:4
#SBATCH -N 1
#SBATCH -c 96
#SBATCH --mem=400G
#SBATCH -t 2-00:00:00
#SBATCH --signal=USR1@900
#SBATCH -J parauto-gim-13b
#SBATCH -o /mnt/vast-kisski/projects/hammerhai-parallax/parauto13b/logs/gim_13b_%j.log
set -euo pipefail
K=/mnt/vast-kisski/projects/hammerhai-parallax/parauto13b
set -a; source $K/secrets.env; set +a
export PAIR_ROOT=$K HF_HOME=$K/hf OMP_NUM_THREADS=8
cd $K/code/compare
run="$K/code/vpd/.venv/bin/torchrun --standalone --nproc_per_node=4"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ -f $K/output/gim_13b/plan.json ]; then
  RUN_DEADLINE=$(( $(date +%s) + 170000 )) $run train_gim_13b.py --resume
else
  if [ "${CHAIN_RESUME:-0}" = "1" ]; then echo "chained resume job but the run never started; nothing to resume"; exit 1; fi
  rm -rf $K/output/gim_13b_smoke
  SMOKE_STEPS=20 RUN_DEADLINE=$(( $(date +%s) + 1800 )) $run train_gim_13b.py
  echo SMOKE_PASSED
  RUN_DEADLINE=$(( $(date +%s) + 170000 )) $run train_gim_13b.py
fi
echo GIM_13B_DONE
