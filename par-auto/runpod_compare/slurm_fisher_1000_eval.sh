#!/bin/bash
# Early look: 1,000-step checkpoints, all Fisher K=8 ranked (VPD by CI), same 65,536 tokens, one GPU.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J parauto-fisher-1k
#SBATCH -o logs/parauto_fisher_1k_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/fisher_k4_5000/eval_1000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 1000 --sensor fisher --fisher-k 8 \
    --checkpoint "$PAIR_ROOT/output/fisher_k4_5000/checkpoints/001000/model.pth" --run-tag fisher-k4-1000-fisher-rank --output "$out/fisher_run"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 1000 --sensor fisher --fisher-k 8 \
    --checkpoint "$PAIR_ROOT/output/gim_1h/checkpoints/001000/model.pth" --run-tag gim-p05-1000-fisher-rank --output "$out/gim_run"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 1000 \
    --checkpoint "$PAIR_ROOT/checkpoints/gim_vpd_1000/vpd.pth" --run-tag vpd-1000-h200 --output "$out/vpd"
echo FISHER_1K_EVAL_DONE
