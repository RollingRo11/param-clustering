#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --signal=USR1@90
#SBATCH -J parauto-write
#SBATCH -o logs/parauto_write_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 3300 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/write_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/test_write_attribution.py"
.venv/bin/python -u "$compare/train_write_5000.py"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
    --checkpoint "$out/checkpoints/005000/model.pth" --sensor write --run-tag write-5000-klkeep --output "$out/eval"
# Same-GPU controls: GIM p=.5 (only the score differs) and VPD, both at 5,000 steps.
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 \
    --checkpoint "$PAIR_ROOT/output/gim_1h/checkpoints/005000/model.pth" --sensor gim \
    --run-tag gim-p05-5000-write-control-klkeep --output "$out/control_gim"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 5000 \
    --checkpoint "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" --run-tag vpd-5000-write-control-klkeep --output "$out/vpd"
.venv/bin/python -u "$compare/report_write_5000.py"
