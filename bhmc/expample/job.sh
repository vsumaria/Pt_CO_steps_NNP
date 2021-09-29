#!/bin/bash
#SBATCH --output="slurm.%j.%N.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=8G
#SBATCH --account=cla175
#SBATCH --export=ALL
#SBATCH -t 24:00:00

module load gcc
module load openmpi
module load openblas

python run_bh.py
