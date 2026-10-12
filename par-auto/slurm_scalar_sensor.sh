#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J scalar-sensor
#SBATCH -o logs/parauto_scalar_sensor_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/scalar_sensor_${SLURM_JOB_ID}"
mkdir -p "$root"
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/train.py" --target "$parauto_target" \
  --output "$root/output" --steps 50000 --sensor output_effect
code/.venv/bin/python -u "$parauto_source/train.py" --target "$parauto_target" \
  --output "$root/scalar" --steps 50000 --sensor scalar_task_mse
code/.venv/bin/python -u "$parauto_source/compare_scalar_sensor.py" --target "$parauto_target" \
  --run "$root/scalar" --output-run "$root/output" --reference "$COFAC_DATA/parauto/klkeep_10242938" --output "$root/comparison"
