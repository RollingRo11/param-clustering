#!/bin/bash
# GIM training with learned ownership of rank-one pieces (OWN_RATIO, OWN_ALPHA), 5,000 steps, then KLKeep (Fisher K=8, hard ownership).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-gim-own
#SBATCH -o logs/parauto_gim_own_%j.log
set -euo pipefail
export OWN_RATIO=${OWN_RATIO:-4} OWN_ALPHA=${OWN_ALPHA:-0.2} OWN_HARD=${OWN_HARD:-0}
export GIM_DEADLINE=$(( $(date +%s) + 5400 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/train_gim_own_5000.py"
out="$PAIR_ROOT/output/gim_own${OWN_RATIO}_a$(.venv/bin/python -c "print(f'{float(\"$OWN_ALPHA\"):g}')")$( [ "$OWN_HARD" = 1 ] && echo _st || true )_5000"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sensor fisher --fisher-k 8 \
    --checkpoint "$out/checkpoints/005000/model.pth" --run-tag gim-own${OWN_RATIO}-a${OWN_ALPHA}-hard${OWN_HARD}-5000-fisher-rank --output "$out/eval_fisher_k8"
echo GIM_OWN_DONE
