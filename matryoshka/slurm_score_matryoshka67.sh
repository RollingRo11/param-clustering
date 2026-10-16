#!/bin/bash
# KLKeep (Fisher K=8 ranking, the same 65,536 held-out positions as every other KLKeep) of every saved checkpoint of
# two Matryoshka runs, plus the paired VPD run's 15,000-step checkpoint, for a KLKeep-vs-training-tokens curve.
# Writes RUN/eval_fisher_k8_steps/STEP/klkeep.json and output/evaluations/15000/vpd/klkeep.json; skips existing
# results; nothing goes to W&B.
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J matryoshka67-score
#SBATCH -o logs/matryoshka67_score_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled
compare="$PAIR_ROOT/code/compare"
snap="${MATRYOSHKA_SNAP:-$PAIR_ROOT/code/matryoshka67_budget_20260927}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
exp="$PAIR_ROOT/output/matryoshka67_20260923"
cd "$PAIR_ROOT/code/vpd"
for run in main_hard_lp1 main_hard_lp1_strat_per1_pow1.5; do
  for step in 500 1000 1500 2000 2500 3000 3500 4000 4500; do
    out="$exp/$run/eval_fisher_k8_steps/$step"
    if [ -f "$out/klkeep.json" ]; then continue; fi
    .venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step "$step" --sensor fisher --fisher-k 8 \
        --checkpoint "$exp/$run/checkpoints/$(printf %06d "$step")/model.pth" \
        --run-tag "matryoshka67-$run-$step-fisher-k8-klkeep" --output "$out"
  done
done
out="$PAIR_ROOT/output/evaluations/15000/vpd"
if [ ! -f "$out/klkeep.json" ]; then
  .venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 15000 \
      --checkpoint "$PAIR_ROOT/checkpoints/step_15000/vpd.pth" --output "$out"
fi
echo MATRYOSHKA_SCORE_DONE
