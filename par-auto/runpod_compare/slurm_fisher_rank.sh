#!/bin/bash
# Re-rank existing 5k checkpoints by the Fisher score (eval only, no training).
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -J parauto-fisher-rank
#SBATCH -o logs/parauto_fisher_rank_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/fisher_rank"
gim="$PAIR_ROOT/output/gim_1h/checkpoints/005000/model.pth"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_fisher_attribution.py"
for sensor in fisher fisher_gim; do
  .venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --checkpoint "$gim" \
      --sensor $sensor --fisher-k 8 --run-tag gim-p05-5000-${sensor}-rank --output "$out/gim_p05_${sensor}_k8"
done
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --checkpoint "$gim" \
    --sensor fisher_gim --fisher-k 32 --run-tag gim-p05-5000-fisher_gim-k32-rank --output "$out/gim_p05_fisher_gim_k32"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
    --checkpoint "$PAIR_ROOT/output/write_5000/checkpoints/005000/model.pth" \
    --sensor fisher_gim --fisher-k 8 --run-tag write-5000-fisher_gim-rank --output "$out/write_fisher_gim_k8"
echo FISHER_RANK_DONE
