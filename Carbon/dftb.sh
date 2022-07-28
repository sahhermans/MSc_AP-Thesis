#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 32 
# set max wallclock time
#SBATCH -t 00:45:00
# set name of job
#SBATCH --job-name=dftb_run
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o dftb.out
module purge
module load 2021
module load intel/2021a
#module load CMake/3.20.1-GCCcore-10.3.0
module load ELSI/2.6.4-intel-2021a-PEXSI
srun dftb+ | tee output
