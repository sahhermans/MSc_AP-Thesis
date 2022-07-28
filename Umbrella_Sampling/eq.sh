#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 64
# set max wallclock time
#SBATCH -t 15:00:00
# set memory requirement
# set name of job
#SBATCH --job-name=lammps_total
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o total.out
module load 2021 foss/2021a
#srun lmp_mpi < input.01.lammps
#srun lmp_mpi < input.02.lammps
#cp data.02.lammps data_original.02.lammps
#python datascript.py
#srun lmp_mpi < input.03.lammps
srun lmp_mpi < input.04.lammps
