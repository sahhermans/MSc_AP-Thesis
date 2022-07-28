import pandas as pd
import numpy as np
import h5py

exclude = list(range(0, 30000))
del exclude[14:350]
infiledftb = pd.read_csv("detailed.out",sep = '      ',header =None,engine = 'python', skiprows = exclude)
infiledftb.columns = ['Atom no.','q']
infiledftb = infiledftb.sort_values('Atom no.')

with open('data.04.lammps', 'r') as fin:
    datalammps = fin.read().splitlines(True)
beginning = datalammps[:32]
ending = datalammps[7071:]

exclude = list(range(0, 30000))
del exclude[32:7071]
infilelammps = pd.read_csv("data.04.lammps", skiprows = exclude, sep = ' ', header = None)
infilelammps.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']
                                     
infilelammps = infilelammps.sort_values('Atom no.')
infilelammpsC_a = infilelammps[(infilelammps['Type'] == 1)]
infilelammpsRest = infilelammps[(infilelammps['Type'] != 1)]

infilelammpsC_a = infilelammpsC_a.assign(q = list(infiledftb['q']))

if list(infiledftb['q']) != list(infilelammpsC_a['q']):
    print("ERROR: Copying active charges unsuccessful")

infilelammps = pd.concat([infilelammpsC_a,infilelammpsRest],axis=0)
infilelammpsAsString = infilelammps.to_string(header=False,index=False,float_format='%.16f')

with open('data.practice.lammps', 'w') as fout:
    fout.writelines(infilelammpsAsString)

with open('data.practice.lammps', 'r') as fin:
    practicedata = fin.read().splitlines(True)
for i in range(len(practicedata)):
    practicedata[i] = practicedata[i].lstrip().replace("    ", " ").replace("   ", " ").replace("  ", " ")

with open('data.04.lammps', 'w') as fout:
    fout.writelines(beginning)
    fout.writelines(practicedata)
    fout.writelines('\n')
    fout.writelines(ending)

exclude = list(range(0, 30000))
del exclude[32:7071]
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