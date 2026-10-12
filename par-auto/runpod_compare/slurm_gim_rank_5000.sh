#!/bin/bash
# GIM training with rank-k pieces (VR_RANK, VR_RANK_COEFF), 5,000 steps, then KLKeep with Fisher K=8 whole-piece ranking.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-gim-rank
#SBATCH -o logs/parauto_gim_rank_%j.log
set -euo pipefail
export VR_RANK=${VR_RANK:-8} VR_RANK_COEFF=${VR_RANK_COEFF:-1}
export GIM_DEADLINE=$(( $(date +%s) + 5400 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/train_gim_rank_5000.py"
out="$PAIR_ROOT/output/gim_rank${VR_RANK}_c$(.venv/bin/python -c "print(f'{float(\"$VR_RANK_COEFF\"):g}')")_5000"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sensor fisher --fisher-k 8 \
    --checkpoint "$out/checkpoints/005000/model.pth" --run-tag gim-rank${VR_RANK}-c${VR_RANK_COEFF}-5000-fisher-rank --output "$out/eval_fisher_k8"
echo GIM_RANK_DONE
