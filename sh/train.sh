#! /bin/sh

#PBS -q regular-mig
#PBS -l select=4
#PBS -l walltime=48:00:00
#PBS -W group_list=GROUP
#PBS -j oe

module purge

cd ${PBS_O_WORKDIR}
uv run scripts/train.py
