import pandas as pd
import h5py

exclude = list(range(0, 30000))
del exclude[32:7071]
infile = pd.read_csv("data.041.lammps", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']

infile = infile.sort_values('Atom no.')
infile = infile.drop('nx',1)
infile = infile.drop('ny',1)
infile = infile.drop('nz',1)

f = h5py.File('final_dftb1.h5','a')
new_data = infile.to_numpy()

f['data'].resize((f['data'].shape[0] + 1), axis=0)
f['data'][-1] = new_data

f.close()

exclude = list(range(0, 30000))
del exclude[32:7071]
infile = pd.read_csv("data.042.lammps", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']

infile = infile.sort_values('Atom no.')
infile = infile.drop('nx',1)
infile = infile.drop('ny',1)
infile = infile.drop('nz',1)

f = h5py.File('final_dftb2.h5','a')
new_data = infile.to_numpy()

f['data'].resize((f['data'].shape[0] + 1), axis=0)
f['data'][-1] = new_data

f.close()
