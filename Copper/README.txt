In this folder all files required to run a coupled LAMMPS/DFTB+ 
simulation of an copper electrode-electrolyte interface can be
found. Below, a short description is given of all files.

slako: 						Folder containing relevant Slater-Koster files.
data.04.lammps: 			Example of LAMMPS output data file of given system.
data_extracter.py: 			Saves simulation data to h5py file, to be used for 
							evaluation of results, and traing neural network. 
							Currently unused, replicated in "extract_dftb_data.py"
data_extracter_init.py: 	Initialises h5py file.
datascript.py: 				Script that can be used to automatically shrink 
							simulation box (for more efficient simulation), for 
							example, when moving from input.02 to input.03 or further.
detailed.output				Output file of DFTB+, containing predicted charges.
dftb.sh 					Script to run DFTB+ on a cluster.
dftb_in.hsd					DFTB+ input file.
dftb_in_orig.hsd			DFTB+ input file.
extract_dftb_data.py		Script that transfers relevant data from DFTB+ to 
							LAMMPS.
extract_lammps_data_03.py	Script that transfers relevant data from LAMMPS to 
							DFTB+.
extract_lammps_data_04.py	Script that transfers relevant data from LAMMPS to 
							DFTB+.
H2O.txt						LAMMPS molecule file for water, used in input.01.lammps
input.01.lammps				Initialisation of LAMMPS system.
input.02.lammps				Energy minimisation of LAMMPS system.
input.03.lammps				Run of LAMMPS system.		
input.04.lammps				Short run of LAMMPS system, to be used in coupled loop.
input.MSD.lammps			Run of LAMMPS system, with data dumps for MSD calculation.
mdanalysis.py				Example file of usage of MDAnalysis.
min.sh						Script to run LAMMPS input files on a cluster.
MSD.py						Script used to process MSD data.
paramaters.lammps			LAMMPS parameter file.
run.sh						Script to run LAMMPS/DFTB+ loop on a cluster.
visualisation.sh			VMD script to visualise LAMMPS dump/dcd files.
						