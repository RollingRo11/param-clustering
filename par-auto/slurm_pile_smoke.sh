#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J parauto-pile-smoke
#SBATCH -o logs/parauto_pile_smoke_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/pile_smoke_${SLURM_JOB_ID}"
mkdir -p "$root"
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
common=(--target "$COFAC_DATA/target/model_step_99999.pt" --tokens "$COFAC_DATA/geo67/stream8k/pile_1b_uint16.bin" --cache "$root/cache" --output "$root/run")
code/.venv/bin/python -u "$parauto_source/train_pile.py" cache "${common[@]}" --blocks 4
code/.venv/bin/python -u "$parauto_source/train_pile.py" train "${common[@]}" --max-steps 2
code/.venv/bin/python -u "$parauto_source/train_pile.py" train "${common[@]}" --max-steps 2 --resume
code/.venv/bin/python -u "$parauto_source/train_pile.py" evaluate "${common[@]}" --eval-blocks 1
echo PILE_SMOKE_COMPLETE
