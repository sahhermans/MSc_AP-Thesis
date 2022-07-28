import pandas as pd
import numpy as np

exclude = list(range(0, 30000))
del exclude[24:7064]
infile = pd.read_csv("data.02.lammps", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['A','Molecule no.','Type','q','x','y','z','nx','ny','nz']

# write the copper electrode input structure file
infilegeo = infile[['A','Type','x','y','z']]
infilegeo = infilegeo[(infilegeo['Type'] == 1)]
infilegeo = infilegeo.sort_values('A')
n_list = np.arange(1,337)
infilegeo = infilegeo.assign(A = list(n_list))
infilegeo['Type'] = 1
nC = len(infilegeo)
infilegeoAsString = infilegeo.to_string(header=False,index=False,float_format = '%.16e')
outfilegeoname = "ingeo.gen"

outfilegeo = open(outfilegeoname, 'w')
outfilegeo.write(str(nC) + " S \n")
outfilegeo.write("C \n")
outfilegeo.write(infilegeoAsString + '\n')
outfilegeo.write('0.0000000000E00   0.0000000000E00   0.0000000000E00 \n')
outfilegeo.write('2.9659638028809457E01   0.0000000000E00   0.0000000000E00 \n')
outfilegeo.write('0.0000000000E00   2.9967000000000002E01   0.0000000000E00 \n')
outfilegeo.write('0.0000000000E00   0.0000000000E00   20.00000000E01 \n')
outfilegeo.close()

# write the point charge input file
infilepc = infile[['Type','x','y','z','q']]
infilepc = infilepc[(infilepc['Type'] != 1 )]
infilepc = infilepc.drop('Type',1)
infilepcAsString = infilepc.to_string(header=False,index=False,float_format = '%.16e')
outfilepcname = "inpc.dat"

outfilepc = open(outfilepcname, 'w')
outfilepc.write(infilepcAsString + '\n')
outfilepc.close()
