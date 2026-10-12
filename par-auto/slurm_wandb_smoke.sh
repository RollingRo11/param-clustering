#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -J parauto-wandb
#SBATCH -o logs/parauto_wandb_%j.log
source "${PARAUTO_SOURCE:-/projects/RohanKathuria/param-clustering/code/par-auto-current}/cluster_env.sh"
root="$COFAC_DATA/parauto/wandb_smoke_${SLURM_JOB_ID}"
code/.venv/bin/python -u -m pytest "$parauto_source/tests/test_wandb_logging.py" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/validate_wandb.py" --output "$root"
code/.venv/bin/python -u "$parauto_source/verify_wandb_remote.py" --output "$root"
