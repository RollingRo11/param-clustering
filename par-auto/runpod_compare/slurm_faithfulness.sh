#!/bin/bash
# Are kept components faithful parts of the computation? par-auto 1B vs VPD published (400k) and paired final.
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -J parauto-faithful
#SBATCH -o logs/parauto_faithful_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"
out="$PAIR_ROOT/output/faithfulness"
cd "$PAIR_ROOT/code/vpd"
for keep in 0.01 0.1; do
  .venv/bin/python -u "$compare/faithfulness_probe.py" --keep $keep --sensor fisher \
      --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/parauto_1b_keep${keep}.json"
  .venv/bin/python -u "$compare/faithfulness_probe.py" --keep $keep --method vpd \
      --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/vpd_published_keep${keep}.json"
  .venv/bin/python -u "$compare/faithfulness_probe.py" --keep $keep --method vpd \
      --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" --output "$out/vpd_paired_keep${keep}.json"
done
echo FAITHFULNESS_ALL_DONE
