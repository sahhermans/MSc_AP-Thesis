This folder contains files that can be used for performing
umbrella sampling. Most files provided here are also found
in the copper folder. 

input.04.lammps contains the actual umbrella sampling script. 
In this script, a molecule of choice is fixed to some position
using a thether. Subsequently, it's position is dumped for 
some time. Afterwards, the script loops back and changes the
thether position. 

After having finished the simulation of input.04.lammps, the
octave.m script can be used to create a file accepted by the 
WHAM algorithm, linking all output files. The created metadata
file can then be used by the WHAM algorithm. 

Notice that the harmonic potentials in LAMMPS and in WHAM 
differ by a factor 1/2.