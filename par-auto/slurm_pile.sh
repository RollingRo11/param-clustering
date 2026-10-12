#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J parauto-pile
#SBATCH -o logs/parauto_pile_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="${PILE_RUN:-$COFAC_DATA/parauto/pile_main}"
mkdir -p "$root"
tar -czf "$root/source_${SLURM_JOB_ID}.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
common=(--target "$COFAC_DATA/target/model_step_99999.pt" --tokens "$COFAC_DATA/geo67/stream8k/pile_1b_uint16.bin" --cache "$root/cache" --output "$root/run")
if [[ ! -f "$root/cache/manifest.json" ]]; then
  code/.venv/bin/python -u "$parauto_source/train_pile.py" cache "${common[@]}"
fi
resume=()
if [[ -f "$root/run/checkpoint.pt" ]]; then resume=(--resume); fi
code/.venv/bin/python -u "$parauto_source/train_pile.py" train "${common[@]}" "${resume[@]}"
code/.venv/bin/python -u "$parauto_source/train_pile.py" evaluate "${common[@]}"
