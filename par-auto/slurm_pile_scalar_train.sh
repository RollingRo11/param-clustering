#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH --signal=B:USR1@180
#SBATCH -J parauto-bf16-8h
#SBATCH -o logs/parauto_scalar_train_%j.log
source "${PARAUTO_SOURCE:?Set the immutable source directory}/cluster_env.sh"
root="${PARAUTO_RUN:?Set the experiment root}"
# Forward the advance SLURM signal to Python for an atomic final checkpoint.
child=""
trap 'if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1
code/.venv/bin/python -u "$parauto_source/train_pile_scalar.py" \
  --target "$COFAC_DATA/target/model_step_99999.pt" \
  --tokens "$COFAC_DATA/geo67/stream8k/pile_1b_uint16.bin" \
  --output "$root/train" --profile-result "$root/profile/result.json" --batch 32768 --hours 8 &
child=$!
wait "$child" || { status=$?; if kill -0 "$child" 2>/dev/null; then wait "$child"; else exit "$status"; fi; }
