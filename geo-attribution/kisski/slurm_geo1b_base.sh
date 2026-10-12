#!/bin/bash
# Baseline GIM geo-attribution on Llama-3.2-1B: run1b.sh recipe with --sensor gim
# at C=4096 (Rohan, 2026-10-12). N=32768 keeps 8 positions per cluster, the
# c2048_1b.sh ratio. Proportional softpart, evals at gate 0.02 and 0.005.
# Stages skip when their output exists, so a resubmit resumes.
#SBATCH -A hammerhai-parallax
#SBATCH -p kisski-h100
#SBATCH --gres=gpu:H100:4
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 24:00:00
#SBATCH -J geo1b-base
#SBATCH -o /mnt/vast-kisski/projects/hammerhai-parallax/geo1b/logs/base_%j.log
set -euo pipefail
source /mnt/vast-kisski/projects/hammerhai-parallax/geo1b/code/kisski/env_geo1b.sh
T=gim1b; D=$G/data/$T
E="--tag $T --eval_batches 4 --batch_seqs 4 --seq_len 512"
step () { echo; echo "########## $* ##########"; date; }
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ ! -f $D/collect_rank3.pt ]; then
  step "collect (GIM, N=32768, 4 ranks)"
  $TR geo1b.py collect --tag $T --sensor gim --n_positions 32768 \
    --pos_per_seq 16 --batch_seqs 8 --seq_len 512
fi
if [ ! -f $D/gram.pt ]; then
  step "gram (module-sharded over 4 ranks)"
  $TR geo1b.py gram --tag $T --gram_block 256
fi
if [ ! -f $D/labels_C4096.pt ]; then
  step "factor: spectral + spherical k-means, C=4096"
  CUDA_VISIBLE_DEVICES=0 $PY geo1b.py factor --tag $T --C 4096
fi
if [ ! -f $D/banks_base.pt ]; then
  step "extract_ps (proportional softpart)"
  CUDA_VISIBLE_DEVICES=0 $PY geo1b.py extract_ps --tag $T --C 4096 \
    --soft_T 1.0 --soft_s 8 --banks_tag base
fi
step "eval at gate 0.02 and 0.005"
ln -sf banks_base.pt $D/banks_base_t005.pt
CUDA_VISIBLE_DEVICES=0 $PY geo1b.py eval $E --gate_thresh 0.02 --banks_tag base &
CUDA_VISIBLE_DEVICES=1 $PY geo1b.py eval $E --gate_thresh 0.005 --banks_tag base_t005 &
wait
echo GEO1B_BASE_DONE
