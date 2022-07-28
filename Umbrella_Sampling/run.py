import subprocess

# create lattice
#subprocess.run(["python","crystalline_atoms.py"])

# minimize charged lattice in dftb+ 
#subprocess.run(["dftb+","|","tee output"])

# extract .gen file from output
#subprocess.run(["python","Extract_dftb_data.py"])

# generate system in lammps
subprocess.run(["lmp","-in","input.01.lammps"])

# minimize system in lammps
subprocess.run(["lmp","-in","input.02.lammps"])
subprocess.run(["lmp","-in","input.03.lammps"])

# extract .gen files from output
subprocess.run(["python","Extract_data.py"])

# start loop
for i in range(100):

    # calculate partial charges in dftb+
    subprocess.run(["dftb+","|","tee output"])
    
    # extract partial charges via detailed.out
    subprocess.run(["python","Extract_dftb_data.py"])

    # run some # of steps in lammps
    subprocess.run(["lmp","-in","input.04.lammps"])
    
    # extract .gen files from output
    subprocess.run(["python","Extract_lammps_data.py"])
    
    # delete irrelevant files

# end















