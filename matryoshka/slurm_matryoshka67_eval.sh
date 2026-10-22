#!/bin/bash
# KLKeep of a 5,000-step (MATRYOSHKA_EVAL_STEP) Matryoshka checkpoint on the same 65,536 held-out positions as its matched control,
# ranked by Fisher K=8 (all runs) and, for GIM-score runs, also by GIM; then the paired comparisons are
# written as JSON next to the run (nothing goes to W&B), and the adversarial-masking job is submitted.
# Arguments: RUN_DIR [SENSOR=fisher|gim]
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J matryoshka67-eval
#SBATCH -o logs/matryoshka67_eval_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
snap="${MATRYOSHKA_SNAP:-$PAIR_ROOT/code/matryoshka67_20260923}"   # inherited from the training job that submits this
out="${1:-$PAIR_ROOT/output/matryoshka67_20260923/main}"     # run directory, e.g. main, main_hard, main_hard_gim
export MATRYOSHKA_OUT="$out" MATRYOSHKA_SENSOR="${2:-fisher}"
export WANDB_MODE=disabled      # KLKeep is not logged to W&B; results stay in eval_*/klkeep.json and comparison*.json
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
cd "$PAIR_ROOT/code/vpd"
step="${MATRYOSHKA_EVAL_STEP:-5000}"      # the checkpoint to evaluate (a fine-tune ends later than 5,000)
ckpt="$out/checkpoints/$(printf %06d "$step")/model.pth"
name="matryoshka67-$(basename "$out")-$step"
if [ "$MATRYOSHKA_SENSOR" = gim ]; then
  .venv/bin/python -u "$PAIR_ROOT/code/compare/evaluate_checkpoint.py" parauto --expected-step "$step" --sensor gim \
      --checkpoint "$ckpt" --run-tag "$name-gim-klkeep" --output "$out/eval_gim"
fi
.venv/bin/python -u "$PAIR_ROOT/code/compare/evaluate_checkpoint.py" parauto --expected-step "$step" --sensor fisher --fisher-k 8 \
    --checkpoint "$ckpt" --run-tag "$name-fisher-k8-klkeep" --output "$out/eval_fisher_k8"
.venv/bin/python -u "$snap/matryoshka67.py" publish
cd /projects/RohanKathuria/param-clustering
hard="${MATRYOSHKA_HARD_PROBE_SNAP:-}"      # also the hard adversary (dropped pieces fully on or off), random and all-off starts
if [ -n "$hard" ]; then
  for init in random off; do
    PROBE_SNAP="$hard" sbatch --parsable --export=ALL "$hard/slurm_adversarial_probe67.sh" "$ckpt" \
      "$PAIR_ROOT/output/adversarial/matryoshka_$(basename "$out")_global_128_hard_$init.json" \
      --scope global --sequences 128 --adversary hard --init "$init" >> "$out/hard_probe_submissions.txt"
  done
fi
if [ "${MATRYOSHKA_EVAL_INLINE:-0}" = 1 ]; then       # set by a training job that runs this script inline
  bash "$snap/slurm_adversarial_matryoshka67.sh" "$out"
  echo MATRYOSHKA_EVAL_DONE
else
  sbatch --parsable "$snap/slurm_adversarial_matryoshka67.sh" "$out" > "$out/adversarial_submission.txt"
  echo "MATRYOSHKA_EVAL_DONE adversarial job $(cat "$out/adversarial_submission.txt")"
fi
