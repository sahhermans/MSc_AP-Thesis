import pandas as pd
import numpy as np
import h5py

# read DFTB+ charge data
exclude = list(range(0, 30000))
del exclude[14:142]
infiledftb = pd.read_csv("detailed.out",sep = '      ',header =None,engine = 'python', skiprows = exclude)
infiledftb.columns = ['Atom no.','q']

# read LAMMPS all data, except for positionaldata
with open('data.04.lammps', 'r') as fin:
    datalammps = fin.read().splitlines(True)
beginning = datalammps[:32]
ending = datalammps[6902:]

# read positional LAMMPS data
exclude = list(range(0, 30000))
del exclude[32:6902]
infilelammps = pd.read_csv("data.04.lammps", skiprows = exclude, sep = ' ', header = None)
infilelammps.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']
                               
# separate data into front layer cathode (active, a), rest cathode (inactive, ia) and non-cathode (rest)                               
infilelammps = infilelammps.sort_values('Atom no.')
infilelammpsCu_a = infilelammps[(infilelammps['Type'] == 5)]
infilelammpsCu_a = infilelammpsCu_a[infilelammpsCu_a['z'] == max(infilelammpsCu_a['z'])]
infilelammpsCu_ia = infilelammps[(infilelammps['Type'] == 5)]
infilelammpsCu_ia = infilelammpsCu_ia[infilelammpsCu_ia['z'] != max(infilelammpsCu_ia['z'])]
infilelammpsRest = infilelammps[(infilelammps['Type'] != 5)]

# assign DFTB+ computed charges to LAMMPS data
zeros = np.zeros(shape=(256))
infilelammpsCu_a = infilelammpsCu_a.assign(q = list(infiledftb['q']))
infilelammpsCu_ia = infilelammpsCu_ia.assign(q = list(zeros))

if list(infiledftb['q']) != list(infilelammpsCu_a['q']):
    print("ERROR: Copying active charges unsuccessful")
if list(zeros) != list(infilelammpsCu_ia['q']):
    print("ERROR: Copying inactive charges unsuccessful")

infilelammps = pd.concat([infilelammpsCu_a,infilelammpsCu_ia,infilelammpsRest],axis=0)

# write new data to new LAMMPS data file
infilelammpsAsString = infilelammps.to_string(header=False,index=False,float_format='%.16f')
practicedata = infilelammpsAsString.splitlines(True)
for i in range(len(practicedata)):
    practicedata[i] = practicedata[i].lstrip().replace("    ", " ").replace("   ", " ").replace("  ", " ")

with open('data.04.lammps', 'w') as fout:
    fout.writelines(beginning)
    fout.writelines(practicedata)
    fout.writelines('\n')
    fout.writelines('\n')
    fout.writelines(ending)
    
# save relevant data to h5py file, which can be used to evaluate data and train neural network
exclude = list(range(0, 30000))
del exclude[32:6902]
infile = pd.read_csv("data.04.lammps", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']

infile = infile.sort_values('Atom no.')
infile = infile.drop('nx',1)
infile = infile.drop('ny',1)
infile = infile.drop('nz',1)

f = h5py.File('final_dftb.h5','a')
new_data = infile.to_numpy()

f['data'].resize((f['data'].shape[0] + 1), axis=0)
f['data'][-1] = new_data

f.close()