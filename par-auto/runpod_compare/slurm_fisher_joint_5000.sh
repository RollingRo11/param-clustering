#!/bin/bash
# Single-job version (QOS submit cap): profile K=1/2/4/8, train K=4 to 5,000, then KLKeep by Fisher K=8 and GIM.
# If the time budget stops training early, eval is skipped; run slurm_fisher_5000_finish.sh afterwards.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-fisher-joint
#SBATCH -o logs/parauto_fisher_joint_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 6000 )) FISHER_K=4
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/fisher_k4_joint_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_joint.py"
.venv/bin/python -u "$compare/train_fisher_joint_5000.py" || echo TRAINING_STOPPED_EARLY
if [ "$(.venv/bin/python -c "import json;print(json.load(open('$out/latest_checkpoint.json'))['step'])")" = "5000" ]; then
  .venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
      --checkpoint "$out/checkpoints/005000/model.pth" --sensor fisher --fisher-k 8 --run-tag fisher-k4-joint-5000-fisher-k8-klkeep --output "$out/eval_fisher_k8"
  .venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
      --checkpoint "$out/checkpoints/005000/model.pth" --sensor gim --run-tag fisher-k4-joint-5000-gim-klkeep --output "$out/eval_gim"
  echo JOINT_TRAIN_EVAL_DONE
fi
