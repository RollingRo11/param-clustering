#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -J parauto-graph
#SBATCH -o logs/parauto_graph_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
compare="$PAIR_ROOT/code/compare"; out="$PAIR_ROOT/output/graph"
cd "$PAIR_ROOT/code/vpd"
for n in 256; do
  .venv/bin/python -u "$compare/graph_probe.py" --nodes $n --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/parauto.pth" --output "$out/parauto_1b_n${n}.json"
  .venv/bin/python -u "$compare/graph_probe.py" --nodes $n --method vpd --checkpoint "$PAIR_ROOT/checkpoints/published_vs_finished/vpd.pth" --output "$out/vpd_published_n${n}.json"
  .venv/bin/python -u "$compare/graph_probe.py" --nodes $n --method vpd --checkpoint "$PAIR_ROOT/checkpoints/vpd_final/vpd.pth" --output "$out/vpd_paired_n${n}.json"
done
echo GRAPH_ALL_DONE
