#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH --time-min=00:45:00
#SBATCH -J parauto-bf16-profile
#SBATCH -o logs/parauto_scalar_profile_%j.log
source "${PARAUTO_SOURCE:?Set the immutable source directory}/cluster_env.sh"
root="${PARAUTO_RUN:?Set the experiment root}"
mkdir -p "$root/profile"
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C "$parauto_source" .
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/train_pile_scalar.py" \
  --target "$COFAC_DATA/target/model_step_99999.pt" \
  --tokens "$COFAC_DATA/geo67/stream8k/pile_1b_uint16.bin" \
  --output "$root/profile" --steps 1000 --batch 32768 --hours 1
