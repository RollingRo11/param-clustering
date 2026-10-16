#!/bin/bash
# Adversarial masking of a Matryoshka run's 5,000-step checkpoint, as VPD's eval (adversarial_probe.py --scope global
# --sequences 128: one value r in [0, 1] per dropped piece over all 128 x 512 held-out positions, sign ascent on the KL,
# step 0.1; KL after 0-320 steps), kept sets = the top 0.5% / 1% / 10% by Fisher K=8. Same settings as the earlier
# output/adversarial/matryoshka_main_hard*_global_128.json. Skips an existing result.  Arguments: RUN_DIR
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J matryoshka67-adv
#SBATCH -o logs/matryoshka67_adv_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_MODE=disabled      # evals stay off W&B; results stay in output/adversarial/*.json
snap="${MATRYOSHKA_SNAP:-$PAIR_ROOT/code/matryoshka67_20260923}"
(cd "$snap" && sha256sum --check --quiet SHA256SUMS && sha256sum --check --quiet DEPENDENCIES.sha256)
run="$1"
out="$PAIR_ROOT/output/adversarial/matryoshka_$(basename "$run")_global_128.json"
if [ -f "$out" ]; then echo "EXISTS $out"; exit 0; fi
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$PAIR_ROOT/code/compare/adversarial_probe.py" --scope global --sequences 128 \
    --checkpoint "$run/checkpoints/005000/model.pth" --output "$out"
echo MATRYOSHKA_ADVERSARIAL_DONE
