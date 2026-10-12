#!/bin/bash
# Archive the paired-run training outputs (everything in output/ except matryoshka67_20260923) to R2 with r2_archive.py:
# upload with sha256 metadata, re-download and hash every object, list verified checkpoint/score files that can be
# deleted. Deletes nothing. Lives in the home directory while /projects refuses writes (quota).
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH -t 08:00:00
#SBATCH -J r2-archive
#SBATCH -o /home/kathuria.r/r2_archive/logs/archive_%j.log
set -euo pipefail
source /projects/RohanKathuria/param-clustering/env.sh
cd /home/kathuria.r/r2_archive
/projects/RohanKathuria/param-clustering/code/.venv/bin/python -u r2_archive.py \
    /projects/RohanKathuria/param-clustering/data/parauto/paired_1b_2000_20260910/output \
    parauto/paired_1b_2000_20260910/output /home/kathuria.r/r2_archive/output_20261001 \
    --exclude matryoshka67_20260923 --suffixes .pt .pth .npy .npz .safetensors .bin .ckpt
