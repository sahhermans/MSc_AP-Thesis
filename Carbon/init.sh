#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 64
# set max wallclock time
#SBATCH -t 04:00:00
# set name of job
#SBATCH --job-name=init_min_lammps
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o init_min.out
module purge
module load 2021
module load intel/2021a
#module load CMake/3.20.1-GCCcore-10.3.0
#module load FFmpeg/4.3.2-GCCcore-10.3.0
module load ELSI/2.6.4-intel-2021a-PEXSI
module load Python/3.9.5-GCCcore-10.3.0
srun lmp_mpi < input.01.lammps
srun lmp_mpi < input.02.lammps