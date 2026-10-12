#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J parauto-profile
#SBATCH -o logs/parauto_profile_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/profile_${SLURM_JOB_ID}"
for mode in method spd; do
  code/.venv/bin/python -u "$parauto_source/profile_spd.py" --mode "$mode" --target "$parauto_target" \
    --assets "$parauto_assets" --output "$root/$mode"
done
code/.venv/bin/python -u "$parauto_source/analyze_spd_profile.py" --root "$root"
