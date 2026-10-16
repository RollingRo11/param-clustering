#!/bin/bash
# Tiny end-to-end check of every stage and every factor variant on 1 GPU.
#SBATCH -A hammerhai-parallax
#SBATCH -p kisski-h100
#SBATCH --gres=gpu:H100:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 00:45:00
#SBATCH -J geo1b-smoke
#SBATCH -o /mnt/vast-kisski/projects/hammerhai-parallax/geo1b/logs/smoke_%j.log
set -euo pipefail
source /mnt/vast-kisski/projects/hammerhai-parallax/geo1b/code/kisski/env_geo1b.sh
rm -rf $G/data/smoke
nvidia-smi --query-gpu=name,memory.total --format=csv
$PY geo1b.py collect --tag smoke --sensor gim --n_positions 64 --pos_per_seq 16 --batch_seqs 4 --seq_len 512
$PY geo1b.py gram --tag smoke
$PY geo1b.py factor --tag smoke --C 4
$PY geo1b.py factor --tag smoke --C 4 --proj twonn --labels_tag twonn
$PY geo1b.py factor --tag smoke --C 4 --cluster hdbscan --hdb_min_size 2 --labels_tag hdbscan
$PY geo1b.py factor --tag smoke --C 4 --cluster adp --labels_tag adp
for v in base adp; do
  L=""; [ $v != base ] && L="--labels_tag $v"
  $PY geo1b.py extract_ps --tag smoke --C 4 $L --soft_T 1.0 --soft_s 2 --banks_tag $v
  $PY geo1b.py eval --tag smoke --banks_tag $v --gate_thresh 0.02 --eval_batches 1 --batch_seqs 2 --seq_len 128
done
echo SMOKE_PASSED
