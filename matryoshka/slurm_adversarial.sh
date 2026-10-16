#!/bin/bash
# Adversarial masking (VPD paper Sec 3.2 style) for VPD and par-auto. SCOPE=position|shared.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J parauto-adv
#SBATCH -o logs/parauto_adv_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/adversarial"
mkdir -p "$out"
cd "$PAIR_ROOT/code/vpd"
SCOPE=${SCOPE:-position}
.venv/bin/python -u "$compare/adversarial_probe.py" --scope $SCOPE --sequences ${SEQ:-32} --method vpd --gates native,topk \
    --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/vpd_published_${SCOPE}${SEQ:+_$SEQ}.json"
.venv/bin/python -u "$compare/adversarial_probe.py" --scope $SCOPE --sequences ${SEQ:-32} \
    --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/parauto_1b_${SCOPE}${SEQ:+_$SEQ}.json"
.venv/bin/python -u "$compare/adversarial_probe.py" --scope $SCOPE --sequences ${SEQ:-32} --method vpd --gates native,topk \
    --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" --output "$out/vpd_paired_${SCOPE}${SEQ:+_$SEQ}.json"
echo ADVERSARIAL_ALL_DONE
