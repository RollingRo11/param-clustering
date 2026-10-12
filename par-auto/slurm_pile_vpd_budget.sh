#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu-short,gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J parauto-vpd-budget
#SBATCH -o logs/parauto_pile_vpd_budget_%j.log
source /projects/RohanKathuria/param-clustering/code/par-auto-current/cluster_env.sh
root="$COFAC_DATA/parauto/pile_last_vpd_${SLURM_JOB_ID}"
mkdir -p "$root"
tar -czf "$root/source_snapshot.tar.gz" --exclude=out --exclude=__pycache__ -C code par-auto-current
code/.venv/bin/python -u -m pytest "$parauto_source/tests" -q -p no:cacheprovider
code/.venv/bin/python -u "$parauto_source/profile_pile_scalar.py" \
  --target "$COFAC_DATA/target/model_step_99999.pt" --tokens "$COFAC_DATA/geo67/stream8k/pile_1b_uint16.bin" \
  --output "$root/full_model_profile"
common=(--target "$COFAC_DATA/target/model_step_99999.pt" --tokens "$COFAC_DATA/geo67/stream8k/pile_1b_uint16.bin" \
  --cache "$COFAC_DATA/parauto/pile_last_10244800/cache" --steps 5000 --batch 128 --components-up 3072 --components-down 3584)
code/.venv/bin/python -u "$parauto_source/run_pile_last.py" train "${common[@]}" --sensor scalar --output "$root/scalar"
code/.venv/bin/python -u "$parauto_source/run_pile_last.py" evaluate "${common[@]}" --training-sensors scalar --runs "$root" --output "$root/comparison" --all-levels
echo PILE_VPD_BUDGET_COMPLETE
