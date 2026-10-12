#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J spd-short
#SBATCH -o logs/parauto_spd_short_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/spd_short_${SLURM_JOB_ID}"
mkdir -p "$root"
(cd code/spd-paper-reference && sha256sum -c "$parauto_source/spd_reference_manifest.sha256")
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/validate_cleanup.py" --target "$parauto_target" \
  --reference "$PWD/code/par-auto-joint/parauto_objectives.py" --output "$root/projected_parity.json"
for steps in 5000 10000; do
  code/.venv/bin/python -u "$parauto_source/benchmark_spd.py" --mode spd --target "$parauto_target" \
    --assets "$parauto_assets" --output "$root/$steps" --steps "$steps"
done
code/.venv/bin/python -u "$parauto_source/compare_spd_schedules.py" --target "$parauto_target" --spd-runs "$root" \
  --method-runs "$COFAC_DATA/parauto/projected_controls_10243205" --reference "$COFAC_DATA/parauto/klkeep_10242938" \
  --spd-reference-run "$COFAC_DATA/parauto/structured_10242461/spd"
