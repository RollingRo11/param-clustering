#!/bin/bash
# Write-size-trained run at 5,000, ranked by GIM instead of write size (same 65,536 tokens, H200).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J parauto-write-gimrank
#SBATCH -o logs/parauto_write_gimrank_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/write_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
    --checkpoint "$out/checkpoints/005000/model.pth" --sensor gim --run-tag write-5000-gimrank-klkeep --output "$out/eval_gim_rank"
echo WRITE_GIMRANK_DONE
