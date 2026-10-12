#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J parauto-klkeep
#SBATCH -o logs/parauto_klkeep_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/klkeep_${SLURM_JOB_ID}"
mkdir -p "$root"
(cd code/spd-paper-reference && sha256sum -c "$parauto_source/spd_reference_manifest.sha256")
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests/test_klkeep.py" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/evaluate_klkeep.py" --target "$parauto_target" \
  --method-run "$COFAC_DATA/parauto/joint_10242577/joint" \
  --spd-run "$COFAC_DATA/parauto/structured_10242461/spd" --output "$root" --events "${EVENTS:-4096}"
code/.venv/bin/python -u "$parauto_source/report_klkeep.py" --run "$root"
