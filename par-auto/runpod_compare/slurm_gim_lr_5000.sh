#!/bin/bash
# GIM score, peak LR from env GIM_LR (sbatch --export=ALL,GIM_LR=3e-3), 5,000 steps, then KLKeep (Fisher K=8 and GIM ranks).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-gim-lr
#SBATCH -o logs/parauto_gim_lr_%j.log
set -euo pipefail
: "${GIM_LR:?set GIM_LR}"
export GIM_DEADLINE=$(( $(date +%s) + 4800 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/gim_lr${GIM_LR}_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/train_gim_lr_5000.py"
out="$(ls -d $PAIR_ROOT/output/gim_lr*_5000 | while read d; do .venv/bin/python -c "import json,sys;print(sys.argv[1]) if abs(json.load(open(sys.argv[1]+'/plan.json'))['training_config']['lr']-float('$GIM_LR'))<1e-12 else None" "$d"; done | head -1)"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sensor fisher --fisher-k 8 \
    --checkpoint "$out/checkpoints/005000/model.pth" --run-tag gim-lr${GIM_LR}-5000-fisher-rank --output "$out/eval_fisher_k8"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sensor gim \
    --checkpoint "$out/checkpoints/005000/model.pth" --run-tag gim-lr${GIM_LR}-5000-gim-rank --output "$out/eval_gim"
echo GIM_LR_${GIM_LR}_DONE
