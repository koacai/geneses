#! /bin/sh
#PBS -q rt_HC
#PBS -l select=1
#PBS -P gag51394
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}
PYTHONUTF8=1 uv run scripts/preprocess_test.py
