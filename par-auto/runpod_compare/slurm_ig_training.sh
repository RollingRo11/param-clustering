#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p sharing
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=USR1@300
#SBATCH -J parauto-ig5-retrain
#SBATCH -o logs/parauto_ig5_retrain_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/ig5_retraining"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_integrated_attribution.py"
.venv/bin/python -u "$compare/train_ig_comparison.py"
steps=$(.venv/bin/python -c 'import json,os; print(json.load(open(os.environ["PAIR_ROOT"]+"/output/ig5_retraining/plan.json"))["steps"])')
for training in gradient ig5; do
  for sensor in gradient ig; do
    .venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step "$steps" \
      --checkpoint "$out/$training/model.pth" --output "$out/eval/${training}_train_${sensor}_eval" \
      --sensor "$sensor" --ig-k 5 --run-tag "ig5-retrain-${training}-rank-${sensor}"
  done
done
.venv/bin/python -u "$compare/report_ig_training.py"
