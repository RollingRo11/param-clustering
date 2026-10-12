#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J parauto-benchmark
#SBATCH -o logs/parauto_benchmark_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/benchmark_${SLURM_JOB_ID}"
mkdir -p "$root"
(cd code/spd-paper-reference && sha256sum -c "$parauto_source/spd_reference_manifest.sha256")
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
nvidia-smi > "$root/gpu.txt"
for mode in method spd; do
  code/.venv/bin/python -u "$parauto_source/benchmark_spd.py" --mode "$mode" --target "$parauto_target" \
    --assets "$parauto_assets" --output "$root/$mode" --steps "${STEPS:-50000}" --evaluate
done
if [[ "${STEPS:-50000}" == 50000 ]]; then
  code/.venv/bin/python -u "$parauto_source/report_spd_benchmark.py" --full "$root" \
    --released "$COFAC_DATA/parauto/spd_pilot_10241633/released" --output "$root/comparison"
fi
