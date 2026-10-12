#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J parauto-fp16
#SBATCH -o logs/parauto_precision_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/fp16_${SLURM_JOB_ID}"
mkdir -p "$root"
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/validate_cleanup.py" --target "$parauto_target" \
  --reference "$PWD/code/par-auto-joint/parauto_objectives.py" --output "$root/fp32_parity.json"
code/.venv/bin/python -u "$parauto_source/train.py" --target "$parauto_target" --output "$root" --steps 50000 --precision fp16
code/.venv/bin/python -u "$parauto_source/compare_precision.py" --target "$parauto_target" --run "$root" \
  --reference "$COFAC_DATA/parauto/klkeep_10242938"
