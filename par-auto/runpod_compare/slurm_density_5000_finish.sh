#!/bin/bash
# Part 2 (afterok): resume to 5,000 if part 1 ran out of time, then KLKeep ranked by Fisher K=8.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-density-2
#SBATCH -o logs/parauto_density_%j.log
set -euo pipefail
export FISHER_K=4 DENSITY_BETA=${DENSITY_BETA:-0.5}
export GIM_DEADLINE=$(( $(date +%s) + 6000 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/density_b${DENSITY_BETA}_5000"
cd "$PAIR_ROOT/code/vpd"
if [ "$(.venv/bin/python -c "import json;print(json.load(open('$out/latest_checkpoint.json'))['step'])")" != "5000" ]; then
  .venv/bin/python -u "$compare/train_density_5000.py" --resume
fi
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sensor fisher --fisher-k 8 \
    --checkpoint "$out/checkpoints/005000/model.pth" --run-tag density-b${DENSITY_BETA}-5000-fisher-k8-klkeep \
    --output "$out/eval_fisher_k8"
echo DENSITY_TRAIN_DONE
