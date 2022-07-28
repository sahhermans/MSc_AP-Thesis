#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 16 
# set max wallclock time
#SBATCH -t 00:30:00
# set memory requirement
#SBATCH --mem=24000
# set name of job
#SBATCH --job-name=dftb_run
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o dftb.out
mpirun dftb+ | tee output
