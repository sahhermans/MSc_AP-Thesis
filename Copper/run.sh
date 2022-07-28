#!/bin/bash
# set the number of nodes
#SBATCH -N 1
#SBATCH -n 28
# set max wallclock time
#SBATCH --mem=50000
#SBATCH -t 12:00:00
# set name of job
#SBATCH --job-name=run_lammps_dftbplus
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
# send mail to this address
#SBATCH --mail-user=s.a.h.hermans@student.tudelft.nl
# run the application
#SBATCH -o run.out
python extract_lammps_data_03.py
python data_extracter_init.py
for i in {1..1000}
do 
	mpirun dftb+ | tee output
	python extract_dftb_data.py
	mpirun lmp_mpi < input.04.lammps
	cat dump.04.lammpstrj >> dumps.04.lammpstrj
	cat dumpq.04.lammpstrj >> dumpsq.04.lammpstrj
	python extract_lammps_data_04.py
done
