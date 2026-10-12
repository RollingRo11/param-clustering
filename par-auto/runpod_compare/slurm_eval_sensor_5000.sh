#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p sharing
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -J normal-gim-vpd-5k
#SBATCH -o logs/normal_gim_vpd_5k_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/evaluations/gim_normal_vpd_5000"
gim_checkpoint="$PAIR_ROOT/output/gim_1h/checkpoints/005000/model.pth"
# Model exports are renamed atomically only once fully written.
for attempt in {1..60}; do
    if test -s "$gim_checkpoint"; then break; fi
    sleep 2
done
test -s "$gim_checkpoint"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
    --checkpoint "$gim_checkpoint" --sensor gim --run-tag gim-5000-klkeep --output "$out/gim"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
    --checkpoint "$PAIR_ROOT/checkpoints/step_5000/parauto.pth" --sensor gradient \
    --run-tag normal-parauto-5000-klkeep --output "$out/normal"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 5000 \
    --checkpoint "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" --run-tag vpd-5000-klkeep --output "$out/vpd"
.venv/bin/python -u "$compare/report_gim_normal_vpd_5000.py"
