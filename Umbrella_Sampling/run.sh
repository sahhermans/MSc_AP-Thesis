#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 28
# set max wallclock time
#SBATCH -t 24:00:00
# set memory requirement
#SBATCH --mem=42000
# set name of job
#SBATCH --job-name=run_lammps_dftb+
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o run.out
python extract_lammps_data_03.py
mv data.03.lammps data.04.lammps
for i in {1..400}
do 
	mpirun dftb+ | tee output
	python extract_dftb_data.py
	mpirun lmp_mpi < input.04.lammps
	python extract_lammps_data_04.py
	python data_ensembler.py
done