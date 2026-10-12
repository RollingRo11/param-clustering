#!/bin/bash
#SBATCH -A n.saad
#SBATCH -p short
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -J parauto-ig5-stage
#SBATCH -o logs/parauto_ig5_stage_%j.log
set -euo pipefail
cd /projects/RohanKathuria/param-clustering
source env.sh
export PAIR_ROOT="$PWD/data/parauto/paired_1b_2000_20260910"
cd "$PAIR_ROOT/code/vpd"
.venv/bin/python -u -c 'import sys; sys.path.insert(0,"../compare"); from stage_milestone import download; from common import ROOT,R2_PREFIX; download(R2_PREFIX+"/data/train.uint16",ROOT/"data/train.uint16")'
