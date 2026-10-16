#!/bin/bash
# Reproduce VPD's published-run training streams (8 ranks x 3.2M rows x 513 tokens) for the 13B GIM run.
# Genuinely short job, so the 2h QOS. One A100 is the smallest allocation the kisski partitions allow.
#SBATCH -A hammerhai-parallax
#SBATCH -p kisski
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=2h
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -J vpd-stream-prep
#SBATCH -o /mnt/vast-kisski/projects/hammerhai-parallax/parauto13b/logs/vpd_stream_prep_%j.log
set -euo pipefail
K=/mnt/vast-kisski/projects/hammerhai-parallax/parauto13b
set -a; source $K/secrets.env; set +a
export PAIR_ROOT=$K HF_HOME=$K/hf OMP_NUM_THREADS=2
cd $K/code/compare
py=$K/code/vpd/.venv/bin/python
for r in 0 1 2 3 4 5 6 7; do
  $py -u prepare_vpd_stream.py --out $K/data/vpd_stream --rank $r > $K/logs/vpd_stream_rank${r}_${SLURM_JOB_ID}.log 2>&1 &
done
fail=0; for pid in $(jobs -p); do wait $pid || fail=1; done
if [ $fail -ne 0 ]; then echo "A RANK FAILED"; tail -n 5 $K/logs/vpd_stream_rank*_${SLURM_JOB_ID}.log; exit 1; fi
$py -u prepare_vpd_stream.py --out $K/data/vpd_stream --finalize
echo VPD_STREAM_READY
