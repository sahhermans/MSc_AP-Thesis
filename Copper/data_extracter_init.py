# initiate h5py file
import h5py

f = h5py.File('final_dftb.h5','a')

dset = f.create_dataset('data', (0,6869,7), maxshape=(None,6869,7), compression="gzip", compression_opts=9)

f.close()