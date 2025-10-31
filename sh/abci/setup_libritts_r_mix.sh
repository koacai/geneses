#! /bin/sh
#PBS -q rt_HC
#PBS -l select=1
#PBS -P gag51394
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}
uv run scripts/setup_libritts_r_mix.py
