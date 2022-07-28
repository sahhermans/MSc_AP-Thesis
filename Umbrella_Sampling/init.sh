#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 4
# set max wallclock time
#SBATCH -t 00:30:00
# set memory requirement
#SBATCH -o test_ini.out
#SBATCH --mem=6000
# set name of job
#SBATCH --job-name=lammps_init
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o init.out
mpirun lmp_mpi < input.01.lammps
