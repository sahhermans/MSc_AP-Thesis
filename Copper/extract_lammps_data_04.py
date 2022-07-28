import pandas as pd
import numpy as np

# read in data
exclude = list(range(0, 30000))
del exclude[32:6902]
infile = pd.read_csv("data.04.lammps", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['A','Molecule no.','Type','q','x','y','z','nx','ny','nz']

# select copper electrode data and change it to DFTB+ input format
infilegeo = infile[['A','Type','x','y','z']]                    
infilegeo = infilegeo[(infilegeo['Type'] == 5)]                 # select copper cathode
infilegeo = infilegeo[infilegeo['z'] == max(infilegeo['z'])]    # only front layer
infilegeo = infilegeo.sort_values('A')                          # sort to keep same order at all times
n_list = np.arange(1,129)                                       # change number and types to DFTB+ ones
infilegeo = infilegeo.assign(A = list(n_list))
infilegeo['Type'] = 1

# write the copper electrode input structure file
nCu = len(infilegeo)
infilegeoAsString = infilegeo.to_string(header=False,index=False,float_format = '%.16e')
outfilegeoname = "ingeo.gen"

outfilegeo = open(outfilegeoname, 'w')
outfilegeo.write(str(nCu) + " S \n")
outfilegeo.write("Cu \n")
outfilegeo.write(infilegeoAsString + '\n')
outfilegeo.write('0.0000000000E00   0.0000000000E00   0.0000000000E00 \n')
outfilegeo.write('2.8919200000E01   0.0000000000E00   0.0000000000E00 \n')
outfilegeo.write('0.0000000000E00   2.8919200000E01   0.0000000000E00 \n')
outfilegeo.write('0.0000000000E00   0.0000000000E00   21.689400000E01 \n')
outfilegeo.close()

# write the point charge input file
infilepc = infile[['Type','x','y','z','q']]
infilepc = infilepc[(infilepc['Type'] != 5 )]                   # all atoms not in cathode are modelled as point charges
infilepc = infilepc.drop('Type',1)
infilepcAsString = infilepc.to_string(header=False,index=False,float_format = '%.16e')
outfilepcname = "inpc.dat"

outfilepc = open(outfilepcname, 'w')
outfilepc.write(infilepcAsString + '\n')
outfilepc.close()
