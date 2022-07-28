In this folder all files required to run a coupled LAMMPS/DFTB+ 
simulation of an carbon electrode-electrolyte interface can be
found. The files in this folder are mostly the same as the ones
in the copper folder, so for a description of the files I will 
refer to that folder. 

I believe the only difference between the carbon and copper 
simulations is the way in which the electrodes was generated.
In the case of copper, LAMMPS was used. For carbon (graphene),
VMD was used. VMD > Extensions > Modelling > Nanotube Builder.
The generated data file was subsequently processed and 
incorporated into the LAMMPS simulation (I believe) using the
graphenegen.py script. 