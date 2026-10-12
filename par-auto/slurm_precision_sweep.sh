#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J precision-sweep
#SBATCH -o logs/parauto_precision_sweep_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/precision_sweep_${SLURM_JOB_ID}"
mkdir -p "$root"
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
for variant in baseline parameters_fp16 activations_fp16 projection_fp32; do
  code/.venv/bin/python -u "$parauto_source/profile_precision_sweep.py" --target "$parauto_target" \
    --output "$root/$variant" --variant "$variant" --steps 5000 --activation-scope "${ACTIVATION_SCOPE:-cache_and_intermediates}"
done
code/.venv/bin/python -u "$parauto_source/report_precision_sweep.py" "$root"
