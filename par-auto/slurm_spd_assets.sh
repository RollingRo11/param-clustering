#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -J parauto-spd-assets
#SBATCH -o logs/parauto_spd_assets_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
root="$COFAC_DATA/parauto/spd_assets_${SLURM_JOB_ID}"
mkdir -p "$root"
(cd code/par-auto-current && sha256sum -c SHA256SUMS)
cp code/par-auto-current/SHA256SUMS "$root/source.sha256"
code/.venv/bin/python -u code/par-auto-current/spd_assets.py --output "$root"
# Keep the shared Torch installation unchanged. Only install import dependencies
# needed by the authors' residual-MLP code, in this run's own directory.
uv pip install --python code/.venv/bin/python --target "$root/deps" \
  'pydantic>=2,<3' 'wandb>=0.19,<0.20' 'einops>=0.8,<0.9' 'jaxtyping>=0.2,<0.4' \
  'matplotlib>=3.9,<3.11' 'numpy<2.3' pyyaml tqdm python-dotenv fire
echo SPD_DEPENDENCIES_READY
