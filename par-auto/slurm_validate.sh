#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J parauto-validate
#SBATCH -o logs/parauto_validate_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/cleanup_validation_${SLURM_JOB_ID}"
mkdir -p "$root"
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/validate_structured.py" --target "$parauto_target" \
  --pieces "$COFAC_DATA/parauto/spd_full_10241714/B/pieces.pt" --output "$root/equivalence.json"
code/.venv/bin/python -u "$parauto_source/train.py" --target "$parauto_target" --output "$root/method" --steps 32
code/.venv/bin/python -u "$parauto_source/parauto_refine.py" --target "$parauto_target" --run "$root/method" --calibration-events 64 --test-events 64
code/.venv/bin/python -u "$parauto_source/plot_spd_comparison.py" --target "$parauto_target" --run "$root/method" --output "$root/figure10"
code/.venv/bin/python -u "$parauto_source/inspect_components.py" --target "$parauto_target" --run "$root/method" --output "$root/components"
code/.venv/bin/python -u "$parauto_source/inspect_usage.py" --target "$parauto_target" --run "$root/method" --profiles "$root/components" --output "$root/usage"
code/.venv/bin/python -u "$parauto_source/benchmark_spd.py" --mode method --target "$parauto_target" --assets "$parauto_assets" --output "$root/benchmark/method" --steps 32 --evaluate
code/.venv/bin/python -u "$parauto_source/benchmark_spd.py" --mode spd --target "$parauto_target" --assets "$parauto_assets" --output "$root/benchmark/spd" --steps 32 --evaluate
printf 'CLEANUP_VALIDATION_COMPLETE\n'
