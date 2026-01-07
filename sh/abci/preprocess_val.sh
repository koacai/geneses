#! /bin/sh
#PBS -q rt_HC
#PBS -l select=1
#PBS -P gcb50354
#PBS -l walltime=2:00:00
#PBS -J 1-5
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}
PYTHONUTF8=1 uv run scripts/preprocess_val.py \
  +split_idx=$((PBS_ARRAY_INDEX - 1)) \
  +array_num=5
