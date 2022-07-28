import h5py

f = h5py.File('final_dftb.h5','a')

dset = f.create_dataset('data', (0,7039,7), maxshape=(None,7039,7), compression="gzip", compression_opts=9)

f.close()

f = h5py.File('final_dftb1.h5','a')

dset = f.create_dataset('data', (0,7039,7), maxshape=(None,7039,7), compression="gzip", compression_opts=9)

f.close()

f = h5py.File('final_dftb2.h5','a')

dset = f.create_dataset('data', (0,7039,7), maxshape=(None,7039,7), compression="gzip", compression_opts=9)