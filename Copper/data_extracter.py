# append relevant data to h5py file
# currently unused, replicated in "extract_dftb_data.py"
import pandas as pd
import h5py

exclude = list(range(0, 30000))
del exclude[61:7118]
infile = pd.read_csv("data.03.lammps", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']

infile = infile.sort_values('Atom no.')
infile = infile.drop('nx',1)
infile = infile.drop('ny',1)
infile = infile.drop('nz',1)

f = h5py.File('final_dftb.h5','a')
dset = f.create_dataset('data', (1,7056,7), maxshape=(None,7056,7), compression="gzip", chunks = True)

new_data = infile.to_numpy()

f['data'].resize((f['data'].shape[0] + 1), axis=0)
f['data'][-1] = new_data

f.close()

























