#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J parauto-short
#SBATCH -o logs/parauto_short_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/projected_controls_${SLURM_JOB_ID}"
mkdir -p "$root"
for steps in 5000 10000; do
  code/.venv/bin/python -u "$parauto_source/train.py" --target "$parauto_target" --output "$root/$steps" --steps "$steps"
done
