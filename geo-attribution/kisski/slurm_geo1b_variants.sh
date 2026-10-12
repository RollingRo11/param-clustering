#!/bin/bash
# Factor variants on the baseline's collect/gram/eigenpairs, one per GPU:
#   twonn   - projection dim = TwoNN intrinsic dim, spherical k-means C=4096
#   hdbscan - spectral dim 4096, HDBSCAN (min cluster size 8), own cluster count
#   adp     - spectral dim 4096, ADP (dadapy, Z=1.65), own cluster count
# Then the baseline's extraction and evals for each.
#SBATCH -A hammerhai-parallax
#SBATCH -p kisski-h100
#SBATCH --gres=gpu:H100:4
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=900G
#SBATCH -t 24:00:00
#SBATCH -J geo1b-variants
#SBATCH -o /mnt/vast-kisski/projects/hammerhai-parallax/geo1b/logs/variants_%j.log
set -euo pipefail
source /mnt/vast-kisski/projects/hammerhai-parallax/geo1b/code/kisski/env_geo1b.sh
T=gim1b; D=$G/data/$T
E="--tag $T --eval_batches 4 --batch_seqs 4 --seq_len 512"
run_variant () {   # gpu name factor-flags...
  local gpu=$1 v=$2; shift 2
  export CUDA_VISIBLE_DEVICES=$gpu
  [ -f $D/labels_$v.pt ] || $PY geo1b.py factor --tag $T --C 4096 "$@" --labels_tag $v
  [ -f $D/banks_$v.pt ] || $PY geo1b.py extract_ps --tag $T --labels_tag $v \
    --soft_T 1.0 --soft_s 8 --banks_tag $v
  ln -sf banks_$v.pt $D/banks_${v}_t005.pt
  $PY geo1b.py eval $E --gate_thresh 0.02 --banks_tag $v
  $PY geo1b.py eval $E --gate_thresh 0.005 --banks_tag ${v}_t005
  echo "VARIANT_DONE $v"
}
run_variant 0 twonn --proj twonn > $G/logs/variant_twonn_$SLURM_JOB_ID.log 2>&1 &
run_variant 1 hdbscan --cluster hdbscan --hdb_min_size 8 > $G/logs/variant_hdbscan_$SLURM_JOB_ID.log 2>&1 &
run_variant 2 adp --cluster adp --adp_Z 1.65 > $G/logs/variant_adp_$SLURM_JOB_ID.log 2>&1 &
fail=0; for j in $(jobs -p); do wait $j || fail=1; done
tail -n 3 $G/logs/variant_*_$SLURM_JOB_ID.log
[ $fail = 0 ] && echo GEO1B_VARIANTS_DONE
exit $fail
