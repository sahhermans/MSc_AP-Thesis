import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

#%%
Cl = np.load('train_params/Cl_tensor.npy')
K = np.load('train_params/K_tensor.npy')
O2 = np.load('train_params/O2_tensor.npy')
H2 = np.load('train_params/H2_tensor.npy')
q_2D = np.load('train_params/q_unet_tensor.npy')

#%%
Cl_copy = Cl.copy()
K_copy = K.copy()
O2_copy = O2.copy()
H2_copy = H2.copy()
q_2D_copy = q_2D.copy()

#%% Door atom type mee te nemen, had dit allemaal in 1/4 van de code gekund
width_xy = 14.459600448608398
for j in range(13):
    Cl[:,0,:,j] = np.round((Cl_copy[:,0,:,j] + width_xy)  //(3.6149/19.9))
    K[:,0,:,j] = np.round((K_copy[:,0,:,j] + width_xy)    //(3.6149/19.9))
    Cl[:,1,:,j] = np.round((Cl_copy[:,1,:,j] + width_xy)  //(3.6149/19.9))
    K[:,1,:,j] = np.round((K_copy[:,1,:,j] + width_xy)    //(3.6149/19.9))
    O2[:,0,:,j] = np.round((O2_copy[:,0,:,j] + width_xy)  //(3.6149/19.9))
    H2[:,0,:,j] = np.round((H2_copy[:,0,:,j] + width_xy)  //(3.6149/19.9))
    O2[:,1,:,j] = np.round((O2_copy[:,1,:,j] + width_xy)  //(3.6149/19.9))
    H2[:,1,:,j] = np.round((H2_copy[:,1,:,j] + width_xy)  //(3.6149/19.9))

    Cl[:,2,:,j] = Cl_copy[:,2,:,j] - q_2D[0,3,:,j] 
    K[:,2,:,j] = K_copy[:,2,:,j] - q_2D[0,3,:,j] 
    O2[:,2,:,j] = O2_copy[:,2,:,j] - q_2D[0,3,:,j] 
    H2[:,2,:,j] = H2_copy[:,2,:,j] - q_2D[0,3,:,j] 
#%%

H_min = np.min(H2[:,2,:,:12])
Cl[:,2,:,:] = Cl[:,2,:,:] - H_min
K[:,2,:,:] = K[:,2,:,:] - H_min
O2[:,2,:,:] = O2[:,2,:,:] - H_min
H2[:,2,:,:] = H2[:,2,:,:] - H_min

#%%
lim_1 = (3.6149/39.9) * 20
lim_2 = lim_1 + (3.6149/19.9) * 40

for i in range(1000):
    idx_Cl_1 = np.where((Cl[:,2,i] < lim_1) & (Cl[:,2,i] >= 0.0))
    idx_Cl_2 = np.where((Cl[:,2,i] < lim_2) & (Cl[:,2,i] >= lim_1))
    idx_Cl_3 = np.where((Cl[:,2,i] >= lim_2))

    idx_K_1 = np.where((K[:,2,i] < lim_1) & (K[:,2,i] >= 0.0))
    idx_K_2 = np.where((K[:,2,i] < lim_2) & (K[:,2,i] >= lim_1)) 
    idx_K_3 = np.where((K[:,2,i] >= lim_2))
    
    idx_O2_1 = np.where((O2[:,2,i] < lim_1) & (O2[:,2,i] >= 0.0))
    idx_O2_2 = np.where((O2[:,2,i] < lim_2) & (O2[:,2,i] >= lim_1)) 
    idx_O2_3 = np.where((O2[:,2,i] >= lim_2))

    idx_H2_1 = np.where((H2[:,2,i] < lim_1) & (H2[:,2,i] >= 0.0))
    idx_H2_2 = np.where((H2[:,2,i] < lim_2) & (H2[:,2,i] >= lim_1)) 
    idx_H2_3 = np.where((H2[:,2,i] >= lim_2))
    
    Cl[:,2,i][idx_Cl_1] = np.round((Cl[:,2,i][idx_Cl_1])        //(3.6149/39.9))
    Cl[:,2,i][idx_Cl_2] = np.round((Cl[:,2,i][idx_Cl_2] - lim_1)//(3.6149/19.9)) + 20
    Cl[:,2,i][idx_Cl_3] = np.round((Cl[:,2,i][idx_Cl_3] - lim_2)//(3.6149/9.9)) + 60
    
    K[:,2,i][idx_K_1] = np.round((K[:,2,i][idx_K_1])        //(3.6149/39.9))
    K[:,2,i][idx_K_2] = np.round((K[:,2,i][idx_K_2] - lim_1)//(3.6149/19.9)) + 20
    K[:,2,i][idx_K_3] = np.round((K[:,2,i][idx_K_3] - lim_2)//(3.6149/9.9)) + 60
    
    O2[:,2,i][idx_O2_1] = np.round((O2[:,2,i][idx_O2_1])        //(3.6149/39.9))
    O2[:,2,i][idx_O2_2] = np.round((O2[:,2,i][idx_O2_2] - lim_1)//(3.6149/19.9)) + 20
    O2[:,2,i][idx_O2_3] = np.round((O2[:,2,i][idx_O2_3] - lim_2)//(3.6149/9.9)) + 60
    
    H2[:,2,i][idx_H2_1] = np.round((H2[:,2,i][idx_H2_1])        //(3.6149/39.9))
    H2[:,2,i][idx_H2_2] = np.round((H2[:,2,i][idx_H2_2] - lim_1)//(3.6149/19.9)) + 20
    H2[:,2,i][idx_H2_3] = np.round((H2[:,2,i][idx_H2_3] - lim_2)//(3.6149/9.9)) + 60

xy_bins = 160
z_bins = 93
z_bins_H2O = 20

x_Cl = Cl[:,0,:].astype(int)
y_Cl = Cl[:,1,:].astype(int)
z_Cl = Cl[:,2,:].astype(int)

x_K = K[:,0,:].astype(int)
y_K = K[:,1,:].astype(int)
z_K = K[:,2,:].astype(int)

x_O2 = O2[:,0,:].astype(int)
y_O2 = O2[:,1,:].astype(int)
z_O2 = O2[:,2,:].astype(int)

x_H2 = H2[:,0,:].astype(int)
y_H2 = H2[:,1,:].astype(int)
z_H2 = H2[:,2,:].astype(int)

i = 0
idx_Cl = np.where(z_Cl[:,i] < z_bins)
idx_K = np.where(z_K[:,i] < z_bins)
idx_O2 = np.where(z_O2[:,i] < z_bins_H2O)
idx_H2 = np.where(z_H2[:,i] < z_bins_H2O)
    
indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i],z_O2[idx_O2,i],z_H2[idx_H2,i]]),
                    np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i],x_O2[idx_O2,i],x_H2[idx_H2,i]]),
                   np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i],y_O2[idx_O2,i],y_H2[idx_H2,i]])])
    
values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))

s_train = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)
  
for i in range(1,900):
    idx_Cl = np.where(z_Cl[:,i] < z_bins)
    idx_K = np.where(z_K[:,i] < z_bins)
    idx_O2 = np.where(z_O2[:,i] < z_bins_H2O)
    idx_H2 = np.where(z_H2[:,i] < z_bins_H2O)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i],z_O2[idx_O2,i],z_H2[idx_H2,i]]),
                    np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i],x_O2[idx_O2,i],x_H2[idx_H2,i]]),
                   np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i],y_O2[idx_O2,i],y_H2[idx_H2,i]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
    
    s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

i = 900
idx_Cl = np.where(z_Cl[:,i] < z_bins)
idx_K = np.where(z_K[:,i] < z_bins)
idx_O2 = np.where(z_O2[:,i] < z_bins_H2O)
idx_H2 = np.where(z_H2[:,i] < z_bins_H2O)
    
indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i],z_O2[idx_O2,i],z_H2[idx_H2,i]]),
                    np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i],x_O2[idx_O2,i],x_H2[idx_H2,i]]),
                   np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i],y_O2[idx_O2,i],y_H2[idx_H2,i]])])

values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))

s_test = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)
for i in range(901,1000):
    idx_Cl = np.where(z_Cl[:,i] < z_bins)
    idx_K = np.where(z_K[:,i] < z_bins)
    idx_O2 = np.where(z_O2[:,i] < z_bins_H2O)
    idx_H2 = np.where(z_H2[:,i] < z_bins_H2O)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i],z_O2[idx_O2,i],z_H2[idx_H2,i]]),
                    np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i],x_O2[idx_O2,i],x_H2[idx_H2,i]]),
                   np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i],y_O2[idx_O2,i],y_H2[idx_H2,i]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
    
    s_test = torch.cat([s_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

s_train = s_train.coalesce()
s_test = s_test.coalesce()

s_train.requires_grad = True

torch.save(s_train,'s_train_H2O_full')
torch.save(s_train,'s_test_H2O_full')