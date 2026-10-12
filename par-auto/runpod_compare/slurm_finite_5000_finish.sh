#!/bin/bash
# Part 2 of 2: resume from the latest checkpoint to 5,000 updates (no-op if part 1 finished),
# then KLKeep on 8,192 held-out tokens for the finite run, GIM p=.5 and VPD.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --signal=USR1@120
#SBATCH -J parauto-finite-2
#SBATCH -o logs/parauto_finite_%j.log
set -euo pipefail
export GIM_DEADLINE=$(( $(date +%s) + 6300 ))
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/finite_5000"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u "$compare/train_finite_5000.py" --resume
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sequences 16 \
    --checkpoint "$out/checkpoints/005000/model.pth" --sensor finite --run-tag finite-5000-klkeep --output "$out/eval"
# Same-GPU controls on the same 8,192 tokens: GIM p=.5 (score and loss tokens differ) and VPD.
.venv/bin/python -u "$compare/evaluate_checkpoint.py" parauto --expected-step 5000 --sequences 16 \
    --checkpoint "$PAIR_ROOT/output/gim_1h/checkpoints/005000/model.pth" --sensor gim \
    --run-tag gim-p05-5000-finite-control-klkeep --output "$out/control_gim"
.venv/bin/python -u "$compare/evaluate_checkpoint.py" vpd --expected-step 5000 --sequences 16 \
    --checkpoint "$PAIR_ROOT/checkpoints/step_5000/vpd.pth" --run-tag vpd-5000-finite-control-klkeep --output "$out/vpd"
.venv/bin/python -u "$compare/report_finite_5000.py"
