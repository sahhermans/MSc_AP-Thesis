import numpy as np
import pandas as pd
      
bulkfilename = "dump.bulk.lammpstrj"
muurfilename = "dump.muur.lammpstrj"

#%%
n_steps = 25001         # total number of timesteps of 100 fs (25001 * 100 fs = 2.5 ns)
n_blocks = 491          # number of data blocks if data is sliced into 50 ps blocks every 5 ps
n_iter = 50             # every 50 timesteps a new measurement is started (every 5 ps)
n_block_steps = 501     # each block consists of 501 timesteps (50 ps)

n_muur_atoms = 7
n_bulk_atoms = 14

N_muur = n_muur_atoms*n_blocks
N_bulk = n_bulk_atoms*n_blocks

#%% read in data
beg = 9
end = 17
exclude = list(range(0, 600000))
for i in range(n_steps):
    del exclude[beg:end]    
    beg = end + 1
    end = end + 9

infilemuur = pd.read_csv(muurfilename, skiprows = exclude, sep = ' ', header = None)
infilemuur.columns = ['Atom no.','Type','x','y','z']
infilemuur = infilemuur[infilemuur['Atom no.'] != 6798] # 6820 6798 --- 6836 6818 (DFTB)
infilemuur = infilemuur.sort_values('Atom no.', kind  = 'stable').to_numpy()

#%%
i = 0
beg = 9
end = 23

exclude = list(range(0, 600000))
for i in range(n_steps):
    del exclude[beg:end]    
    beg = end - 5
    end = end + 9
    
infilebulk = pd.read_csv(bulkfilename, skiprows = exclude, sep = ' ', header = None)
infilebulk.columns = ['Atom no.','Type','x','y','z']
#infilemuur = infilemuur[infilemuur['Atom no.'] != 6818] # 
infilebulk = infilebulk.sort_values('Atom no.', kind = 'stable').to_numpy()

#%%
msd_muur_x = np.zeros((n_muur_atoms,n_blocks,n_block_steps))
msd_muur_y = np.zeros((n_muur_atoms,n_blocks,n_block_steps))
msd_muur_xy = np.zeros((n_muur_atoms,n_blocks,n_block_steps))

msd_bulk_x = np.zeros((n_bulk_atoms,n_blocks,n_block_steps))
msd_bulk_y = np.zeros((n_bulk_atoms,n_blocks,n_block_steps))
msd_bulk_z = np.zeros((n_bulk_atoms,n_blocks,n_block_steps))
msd_bulk_xy = np.zeros((n_bulk_atoms,n_blocks,n_block_steps))
msd_bulk_xyz = np.zeros((n_bulk_atoms,n_blocks,n_block_steps))

for i in range(n_muur_atoms):
    for k in range(n_blocks):
        for j in range(n_block_steps):
        
            msd_muur_x[i,k,j] = abs(infilemuur[k*n_iter + j + i*n_steps,2] - infilemuur[k*n_iter + i*n_steps,2])**2
            msd_muur_y[i,k,j] = abs(infilemuur[k*n_iter + j + i*n_steps,3] - infilemuur[k*n_iter + i*n_steps,3])**2
            msd_muur_xy[i,k,j] = abs(infilemuur[k*n_iter + j + i*n_steps,2] - infilemuur[k*n_iter + i*n_steps,2])**2 + abs(infilemuur[k*n_iter + j + i*n_steps,3] - infilemuur[k*n_iter + i*n_steps,3])**2

for i in range(n_bulk_atoms):
    for k in range(n_blocks):
        for j in range(n_block_steps):   
            
            msd_bulk_x[i,k,j] = abs(infilebulk[k*n_iter + j + i*n_steps,2] - infilebulk[k*n_iter + i*n_steps,2])**2
            msd_bulk_y[i,k,j] = abs(infilebulk[k*n_iter + j + i*n_steps,3] - infilebulk[k*n_iter + i*n_steps,3])**2
            msd_bulk_z[i,k,j] = abs(infilebulk[k*n_iter + j + i*n_steps,4] - infilebulk[k*n_iter + i*n_steps,4])**2
            msd_bulk_xy[i,k,j] = abs(infilebulk[k*n_iter + j + i*n_steps,2] - infilebulk[k*n_iter + i*n_steps,2])**2 + abs(infilebulk[k*n_iter + j + i*n_steps,3] - infilebulk[k*n_iter + i*n_steps,3])**2
            msd_bulk_xyz[i,k,j] = abs(infilebulk[k*n_iter + j + i*n_steps,2] - infilebulk[k*n_iter + i*n_steps,2])**2 + abs(infilebulk[k*n_iter + j + i*n_steps,3] - infilebulk[k*n_iter + i*n_steps,3])**2 + abs(infilebulk[k*n_iter + j + i*n_steps,4] - infilebulk[k*n_iter + i*n_steps,4])**2

MSD_muur_x = (1/N_muur)*np.sum(np.sum(msd_muur_x,axis = 0),axis=0)
MSD_muur_y = (1/N_muur)*np.sum(np.sum(msd_muur_y,axis = 0),axis=0)
MSD_muur_xy = (1/N_muur)*np.sum(np.sum(msd_muur_xy,axis = 0),axis=0)

MSD_bulk_x = (1/N_bulk)*np.sum(np.sum(msd_bulk_x,axis = 0),axis=0)
MSD_bulk_y = (1/N_bulk)*np.sum(np.sum(msd_bulk_y,axis = 0),axis=0)
MSD_bulk_z = (1/N_bulk)*np.sum(np.sum(msd_bulk_z,axis = 0),axis=0)
MSD_bulk_xy = (1/N_bulk)*np.sum(np.sum(msd_bulk_xy,axis = 0),axis=0)
MSD_bulk_xyz = (1/N_bulk)*np.sum(np.sum(msd_bulk_xyz,axis = 0),axis=0)

#%%
VAR_muur_x = (1/N_muur)*np.sum(np.sum(np.square(msd_muur_x - MSD_muur_x),axis = 0),axis=0)
VAR_muur_y = (1/N_muur)*np.sum(np.sum(np.square(msd_muur_y - MSD_muur_y),axis = 0),axis=0)
VAR_muur_xy = (1/N_muur)*np.sum(np.sum(np.square(msd_muur_xy - MSD_muur_xy),axis = 0),axis=0)

VAR_bulk_x = (1/N_bulk)*np.sum(np.sum(np.square(msd_bulk_x - MSD_bulk_x),axis = 0),axis=0)
VAR_bulk_y = (1/N_bulk)*np.sum(np.sum(np.square(msd_bulk_y - MSD_bulk_y),axis = 0),axis=0)
VAR_bulk_z = (1/N_bulk)*np.sum(np.sum(np.square(msd_bulk_z - MSD_bulk_z),axis = 0),axis=0)
VAR_bulk_xy = (1/N_bulk)*np.sum(np.sum(np.square(msd_bulk_xy - MSD_bulk_xy),axis = 0),axis=0)
VAR_bulk_xyz = (1/N_bulk)*np.sum(np.sum(np.square(msd_bulk_xyz - MSD_bulk_xyz),axis = 0),axis=0)

SE_muur_x = np.sqrt(VAR_muur_x)/np.sqrt(N_muur)
SE_muur_y = np.sqrt(VAR_muur_y)/np.sqrt(N_muur)
SE_muur_xy = np.sqrt(VAR_muur_xy)/np.sqrt(N_muur)

SE_bulk_x = np.sqrt(VAR_bulk_x)/np.sqrt(N_bulk)
SE_bulk_y = np.sqrt(VAR_bulk_y)/np.sqrt(N_bulk)
SE_bulk_z = np.sqrt(VAR_bulk_z)/np.sqrt(N_bulk)
SE_bulk_xy = np.sqrt(VAR_bulk_xy)/np.sqrt(N_bulk)
SE_bulk_xyz = np.sqrt(VAR_bulk_xyz)/np.sqrt(N_bulk)

#%%
import scipy.io
scipy.io.savemat('MSD_muur_x.mat', dict(x=MSD_muur_x))
scipy.io.savemat('MSD_muur_y.mat', dict(x=MSD_muur_y))
scipy.io.savemat('MSD_muur_xy.mat', dict(x=MSD_muur_xy))

scipy.io.savemat('MSD_bulk_x.mat', dict(x=MSD_bulk_x))
scipy.io.savemat('MSD_bulk_y.mat', dict(x=MSD_bulk_y))
scipy.io.savemat('MSD_bulk_z.mat', dict(x=MSD_bulk_z))
scipy.io.savemat('MSD_bulk_xy.mat', dict(x=MSD_bulk_xy))
scipy.io.savemat('MSD_bulk_xyz.mat', dict(x=MSD_bulk_xyz))

scipy.io.savemat('SE_muur_x.mat', dict(x=SE_muur_x))
scipy.io.savemat('SE_muur_y.mat', dict(x=SE_muur_y))
scipy.io.savemat('SE_muur_xy.mat', dict(x=SE_muur_xy))

scipy.io.savemat('SE_bulk_x.mat', dict(x=SE_bulk_x))
scipy.io.savemat('SE_bulk_y.mat', dict(x=SE_bulk_y))
scipy.io.savemat('SE_bulk_z.mat', dict(x=SE_bulk_z))
scipy.io.savemat('SE_bulk_xy.mat', dict(x=SE_bulk_xy))
scipy.io.savemat('SE_bulk_xyz.mat', dict(x=SE_bulk_xyz))

scipy.io.savemat('VAR_muur_x.mat', dict(x=VAR_muur_x))
scipy.io.savemat('VAR_muur_y.mat', dict(x=VAR_muur_y))
scipy.io.savemat('VAR_muur_xy.mat', dict(x=VAR_muur_xy))

scipy.io.savemat('VAR_bulk_x.mat', dict(x=VAR_bulk_x))
scipy.io.savemat('VAR_bulk_y.mat', dict(x=VAR_bulk_y))
scipy.io.savemat('VAR_bulk_z.mat', dict(x=VAR_bulk_z))
scipy.io.savemat('VAR_bulk_xy.mat', dict(x=VAR_bulk_xy))
scipy.io.savemat('VAR_bulk_xyz.mat', dict(x=VAR_bulk_xyz))

#%%
import scipy.io
MSD_muur_x = scipy.io.loadmat('MSD_muur_x.mat')
MSD_muur_y = scipy.io.loadmat('MSD_muur_y.mat')
MSD_muur_xy = scipy.io.loadmat('MSD_muur_xy.mat')

MSD_bulk_x = scipy.io.loadmat('MSD_bulk_x.mat')
MSD_bulk_y = scipy.io.loadmat('MSD_bulk_y.mat')
MSD_bulk_z = scipy.io.loadmat('MSD_bulk_z.mat')
MSD_bulk_xy = scipy.io.loadmat('MSD_bulk_xy.mat')
MSD_bulk_xyz = scipy.io.loadmat('MSD_bulk_xyz.mat')

#%%
MSD_muur_x_peratom = (1/n_blocks)*np.sum(msd_muur_x,axis = 1)
MSD_bulk_x_peratom = (1/n_blocks)*np.sum(msd_bulk_x,axis = 1)

#%%
import matplotlib.pyplot as plt

time = np.linspace(0,n_iter,n_block_steps)

#%%
plt.figure(1)
plt.grid()
#plt.plot(time,MSD_muur)
plt.plot(time[10:],MSD_bulk_x[10:])
plt.plot(time[10:],MSD_bulk_y[10:],linestyle='dashed')
plt.plot(time[10:],MSD_bulk_z[10:],linestyle='dotted')
plt.plot(time[10:],MSD_bulk_xyz[10:],linestyle='dashdot')
plt.fill_between(time[10:], MSD_bulk_x[10:] - SE_bulk_x[10:], MSD_bulk_x[10:] + SE_bulk_x[10:],
                 color='gray', alpha=0.2)
plt.fill_between(time[10:], MSD_bulk_y[10:] - SE_bulk_y[10:], MSD_bulk_y[10:] + SE_bulk_y[10:],
                 color='gray', alpha=0.2)
plt.fill_between(time[10:], MSD_bulk_z[10:] - SE_bulk_z[10:], MSD_bulk_z[10:] + SE_bulk_z[10:],
                 color='gray', alpha=0.2)
plt.fill_between(time[10:], MSD_bulk_xyz[10:] - SE_bulk_xyz[10:], MSD_bulk_xyz[10:] + SE_bulk_xyz[10:],
                 color='gray', alpha=0.2)
plt.xlabel(r'Time [ps]',fontsize=14)
plt.ylabel(r'MSD [$\rm \AA^2$]',fontsize=14)
plt.legend([r'$MSD_x$',r'$MSD_y$',r'$MSD_z$',r'$MSD_{xyz}$'],fontsize=11)
plt.xlim(1,50)
plt.ylim(0,60)
    
#%%
plt.figure(2)
plt.grid()
plt.plot(time[10:],MSD_muur_x[10:])
plt.plot(time[10:],MSD_muur_y[10:],linestyle='dashed')
plt.plot(time[10:],MSD_muur_xy[10:],linestyle='dashdot')
plt.fill_between(time[10:], MSD_muur_x[10:] - SE_muur_x[10:], MSD_muur_x[10:] + SE_muur_x[10:],
                 color='gray', alpha=0.2)
plt.fill_between(time[10:], MSD_muur_y[10:] - SE_muur_y[10:], MSD_muur_y[10:] + SE_muur_y[10:],
                 color='gray', alpha=0.2)
plt.fill_between(time[10:], MSD_muur_xy[10:] - SE_muur_xy[10:], MSD_muur_xy[10:] + SE_muur_xy[10:],
                 color='gray', alpha=0.2)
#plt.plot(time,MSD_bulk)
plt.xlabel(r'Time [ps]',fontsize=14)
plt.ylabel(r'MSD [$\rm \AA^2$]',fontsize=14)
plt.legend([r'$MSD_x$',r'$MSD_y$',r'$MSD_{xy}$'],fontsize=11)
plt.xlim(1,50)
plt.ylim(0)

#%%
plt.figure(2)
plt.grid()
#plt.plot(time[10:],MSD_muur_x[10:])
plt.plot(time[10:],muur_cmd[10:])
plt.plot(time[10:],muur_qmmd[10:],linestyle='dashdot')

#plt.fill_between(time[10:], MSD_muur_x[10:] - SE_muur_x[10:], MSD_muur_x[10:] + SE_muur_x[10:],
              #   color='gray', alpha=0.2)
plt.fill_between(time[10:], muur_qmmd[10:] - SE_muur_qmmd[10:], muur_qmmd[10:] + SE_muur_qmmd[10:],
                 color='gray', alpha=0.2)
plt.fill_between(time[10:], muur_cmd[10:] - SE_muur_cmd[10:], muur_cmd[10:] + SE_muur_cmd[10:],
                 color='gray', alpha=0.2)
#plt.plot(time,MSD_bulk)
plt.xlabel(r'Time [ps]',fontsize=14)
plt.ylabel(r'MSD [$\rm \AA^2$]',fontsize=14)
plt.legend([r'$C-MD$',r'$QM-MD$'],fontsize=11)
plt.xlim(1,50)
plt.ylim(0)

#%%
plt.figure(3)
plt.grid()
#plt.title('MSD vs. time (muur)',fontsize = 14)
for i in range(n_muur_atoms):  
    plt.plot(time,MSD_muur_x_peratom[i,:])
plt.xlabel(r'Time [ps]',fontsize=14)
plt.ylabel(r'MSD [$\rm \AA^2$]',fontsize=14)
#plt.legend([r'$MSD_1$',r'$MSD_2$',r'$MSD_3$',r'$MSD_4$',r'$MSD_5$',r'$MSD_6$',r'$MSD_7$',r'$MSD_8$'],fontsize=11) #,r'$MSD_9$',r'$MSD_{10}$',r'$MSD_{11}$',r'$MSD_{12}$',r'$MSD_{13}$',r'$MSD_{14}$'
plt.xlim(0,50)
plt.ylim(0)

plt.figure(4)
plt.grid()
#plt.title('MSD vs. time (bulk)',fontsize = 14)
for i in range(n_bulk_atoms):
    plt.plot(time,MSD_bulk_x_peratom[i,:])
plt.xlabel(r'Time [ps]',fontsize=14)
plt.ylabel(r'MSD [$\rm \AA^2$]',fontsize=14)
#plt.legend([r'$MSD_1$',r'$MSD_2$',r'$MSD_3$',r'$MSD_4$',r'$MSD_5$',r'$MSD_6$',r'$MSD_7$',r'$MSD_8$',r'$MSD_9$',r'$MSD_{10}$',r'$MSD_{11}$',r'$MSD_{12}$',r'$MSD_{13}$',r'$MSD_{14}$'],fontsize=11)
plt.xlim(0,50)
plt.ylim(0)
