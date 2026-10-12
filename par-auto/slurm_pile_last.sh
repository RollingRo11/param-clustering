#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J parauto-last-mlp
#SBATCH -o logs/parauto_pile_last_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="${PILE_LAST_RUN:-$COFAC_DATA/parauto/pile_last_${SLURM_JOB_ID}}"
mkdir -p "$root"
tar -czf "$root/source_${SLURM_JOB_ID}.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
common=(--target "$COFAC_DATA/target/model_step_99999.pt" --tokens "$COFAC_DATA/geo67/stream8k/pile_1b_uint16.bin" --cache "$root/cache" --steps 5000 --batch 128)
if [[ ! -f "$root/cache/manifest.json" ]]; then
  code/.venv/bin/python -u "$parauto_source/run_pile_last.py" cache "${common[@]}" --output "$root/setup"
fi
for sensor in output scalar; do
  if [[ ! -f "$root/profile-$sensor/result.json" ]]; then
    code/.venv/bin/python -u "$parauto_source/run_pile_last.py" profile "${common[@]}" --sensor "$sensor" --output "$root/profile-$sensor"
  fi
done
for sensor in output scalar; do
  resume=()
  if [[ -f "$root/$sensor/checkpoint.pt" ]]; then resume=(--resume); fi
  code/.venv/bin/python -u "$parauto_source/run_pile_last.py" train "${common[@]}" --sensor "$sensor" --output "$root/$sensor" "${resume[@]}"
done
code/.venv/bin/python -u "$parauto_source/run_pile_last.py" evaluate "${common[@]}" --runs "$root" --output "$root/comparison" --all-levels
echo PILE_LAST_COMPLETE
