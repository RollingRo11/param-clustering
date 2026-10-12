#!/bin/bash
# Per-layer KLKeep: par-auto 1B (GIM + Fisher rank), latest paired VPD, published VPD.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -J parauto-perlayer
#SBATCH -o logs/parauto_perlayer_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/perlayer"
cd "$PAIR_ROOT/code/vpd"
vpd_late=""
for step in 20000 15000 10000; do
  if [ -f "$PAIR_ROOT/checkpoints/step_${step}/vpd.pth" ]; then vpd_late="$PAIR_ROOT/checkpoints/step_${step}/vpd.pth"; echo "paired VPD: step_${step}"; break; fi
done
.venv/bin/python -u "$compare/perlayer_klkeep.py" --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --sensor gim --output "$out/parauto_1b_gim.json"
.venv/bin/python -u "$compare/perlayer_klkeep.py" --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --sensor fisher --output "$out/parauto_1b_fisher.json"
.venv/bin/python -u "$compare/perlayer_klkeep.py" --method vpd --checkpoint "$vpd_late" --output "$out/vpd_paired_late.json"
.venv/bin/python -u "$compare/perlayer_klkeep.py" --method vpd --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/vpd_published.json"
echo PERLAYER_ALL_DONE
