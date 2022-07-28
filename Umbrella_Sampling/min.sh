#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 28
# set max wallclock time
#SBATCH -t 20:00:00
# set memory requirement
#SBATCH --mem=42000
# set name of job
#SBATCH --job-name=lammps_min
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o min.out
mpirun lmp_mpi < input.02.lammps
