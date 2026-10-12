#!/bin/bash
# gim score with VPD's component optimizer schedule + clip 0.01, 5,000 steps, then KLKeep (Fisher K=8 and GIM ranks).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-gim-vpdsched
#SBATCH -o logs/parauto_gim_vpdsched_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 4800 )) FISHER_K=4
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/gim_vpdsched_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/train_gim_vpdsched_5000.py"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sensor fisher --fisher-k 8 \
    --checkpoint "$out/checkpoints/005000/model.pth" --run-tag gim-vpdsched-5000-fisher-rank --output "$out/eval_fisher_k8"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sensor gim \
    --checkpoint "$out/checkpoints/005000/model.pth" --run-tag gim-vpdsched-5000-gim-rank --output "$out/eval_gim"
echo VPDSCHED_gim_DONE
