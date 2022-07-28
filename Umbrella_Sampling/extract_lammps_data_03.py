import pandas as pd

exclude = list(range(0, 30000))
del exclude[33:7090]
infile = pd.read_csv("data.03.lammps", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']

# write the copper electrode input structure file
infilegeo = pd.concat([infile['Atom no.'],infile['Type'],infile['x'],infile['y'],infile['z']], axis = 1)
infilegeo = infilegeo[(infilegeo['Type'] == 10)]
infilegeo = infilegeo.sort_values('Atom no.')
infilegeo['Type'] = 1
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
outfilegeo.write('0.0000000000E00   0.0000000000E00   18.0000000000E01 \n')
outfilegeo.close()

# write the point charge input file
infilepc = pd.concat([infile['Type'],infile['x'],infile['y'],infile['z'],infile['q']], axis = 1)
infilepc = infilepc[(infilepc['Type'] != 10 )]
infilepc = infilepc.drop('Type',1)
infilepcAsString = infilepc.to_string(header=False,index=False,float_format = '%.16e')
outfilepcname = "inpc.dat"

outfilepc = open(outfilepcname, 'w')
outfilepc.write(infilepcAsString + '\n')
outfilepc.close()

