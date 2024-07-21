#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cmake cuda

cmake -S . -B build
cmake --build build -j
cd executable
./SPH
