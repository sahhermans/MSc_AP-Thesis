import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

#%%
Cl = np.load('train_params/Cl_tensor.npy')
K = np.load('train_params/K_tensor.npy')
q_2D = np.load('train_params/q_unet_tensor_sorted.npy')
O2 = np.load('train_params/O2_tensor.npy')
H2 = np.load('train_params/H2_tensor.npy')

Cl_r = np.load('train_params/Cl_tensor_r.npy')
K_r = np.load('train_params/K_tensor_r.npy')
q_2D_r = np.load('train_params/q_tensor_r.npy')
O2_r = np.load('train_params/O2_tensor_r.npy')
H2_r = np.load('train_params/H2_tensor_r.npy')

Cl_xy = np.load('train_params/Cl_tensor_xy.npy')
K_xy = np.load('train_params/K_tensor_xy.npy')
q_2D_xy = np.load('train_params/q_tensor_xy.npy')
O2_xy = np.load('train_params/O2_tensor_xy.npy')
H2_xy = np.load('train_params/H2_tensor_xy.npy')

Cl_minxy = np.load('train_params/Cl_tensor_minxy.npy')
K_minxy = np.load('train_params/K_tensor_minxy.npy')
q_2D_minxy = np.load('train_params/q_tensor_minxy.npy')
O2_minxy = np.load('train_params/O2_tensor_minxy.npy')
H2_minxy = np.load('train_params/H2_tensor_minxy.npy')

Cl = np.stack([Cl,Cl_r,Cl_xy,Cl_minxy],axis=-1)
K = np.stack([K,K_r,K_xy,K_minxy],axis=-1)
O2 = np.stack([O2,O2_r,O2_xy,O2_minxy],axis=-1)
H2 = np.stack([H2,H2_r,H2_xy,H2_minxy],axis=-1)
q_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

Cl_copy = Cl.copy()
K_copy = K.copy()
O2_copy = O2.copy()
H2_copy = H2.copy()
q_2D_copy = q_2D.copy()

#%% Door atom type mee te nemen, had dit allemaal in 1/4 van de code gekund
width_xy = 14.459600448608398
for k in range(4):
    for j in range(13):
        Cl[:,0,:,j,k] = np.round((Cl_copy[:,0,:,j,k] + width_xy)  //(3.6149/19.9))
        K[:,0,:,j,k] = np.round((K_copy[:,0,:,j,k] + width_xy)    //(3.6149/19.9))
        Cl[:,1,:,j,k] = np.round((Cl_copy[:,1,:,j,k] + width_xy)  //(3.6149/19.9))
        K[:,1,:,j,k] = np.round((K_copy[:,1,:,j,k] + width_xy)    //(3.6149/19.9))
        O2[:,0,:,j,k] = np.round((O2_copy[:,0,:,j,k] + width_xy)  //(3.6149/19.9))
        H2[:,0,:,j,k] = np.round((H2_copy[:,0,:,j,k] + width_xy)  //(3.6149/19.9))
        O2[:,1,:,j,k] = np.round((O2_copy[:,1,:,j,k] + width_xy)  //(3.6149/19.9))
        H2[:,1,:,j,k] = np.round((H2_copy[:,1,:,j,k] + width_xy)  //(3.6149/19.9))

        Cl[:,2,:,j,k] = Cl_copy[:,2,:,j,k] - q_2D[0,3,:,j,k] 
        K[:,2,:,j,k] = K_copy[:,2,:,j,k] - q_2D[0,3,:,j,k] 
        O2[:,2,:,j,k] = O2_copy[:,2,:,j,k] - q_2D[0,3,:,j,k] 
        H2[:,2,:,j,k] = H2_copy[:,2,:,j,k] - q_2D[0,3,:,j,k] 
        
#%%
H_min = np.min(H2[:,2,:,:12,:])
Cl[:,2,:,:,:] = Cl[:,2,:,:,:] - H_min
K[:,2,:,:,:] = K[:,2,:,:,:] - H_min
O2[:,2,:,:,:] = O2[:,2,:,:,:] - H_min
H2[:,2,:,:,:] = H2[:,2,:,:,:] - H_min

#%%
lim_1 = (3.6149/39.9) * 20
lim_2 = lim_1 + (3.6149/19.9) * 40

for k in range(4):
    for j in range(13):
        for i in range(1000):
            idx_Cl_1 = np.where((Cl[:,2,i,j,k] < lim_1) & (Cl[:,2,i,j,k] >= 0.0))
            idx_Cl_2 = np.where((Cl[:,2,i,j,k] < lim_2) & (Cl[:,2,i,j,k] >= lim_1))
            idx_Cl_3 = np.where((Cl[:,2,i,j,k] >= lim_2))

            idx_K_1 = np.where((K[:,2,i,j,k] < lim_1) & (K[:,2,i,j,k] >= 0.0))
            idx_K_2 = np.where((K[:,2,i,j,k] < lim_2) & (K[:,2,i,j,k] >= lim_1)) 
            idx_K_3 = np.where((K[:,2,i,j,k] >= lim_2))
            
            idx_O2_1 = np.where((O2[:,2,i,j,k] < lim_1) & (O2[:,2,i,j,k] >= 0.0))
            idx_O2_2 = np.where((O2[:,2,i,j,k] < lim_2) & (O2[:,2,i,j,k] >= lim_1)) 
            idx_O2_3 = np.where((O2[:,2,i,j,k] >= lim_2))

            idx_H2_1 = np.where((H2[:,2,i,j,k] < lim_1) & (H2[:,2,i,j,k] >= 0.0))
            idx_H2_2 = np.where((H2[:,2,i,j,k] < lim_2) & (H2[:,2,i,j,k] >= lim_1)) 
            idx_H2_3 = np.where((H2[:,2,i,j,k] >= lim_2))
            
            Cl[:,2,i,j,k][idx_Cl_1] = np.round((Cl[:,2,i,j,k][idx_Cl_1])        //(3.6149/39.9))
            Cl[:,2,i,j,k][idx_Cl_2] = np.round((Cl[:,2,i,j,k][idx_Cl_2] - lim_1)//(3.6149/19.9)) + 20
            Cl[:,2,i,j,k][idx_Cl_3] = np.round((Cl[:,2,i,j,k][idx_Cl_3] - lim_2)//(3.6149/9.9)) + 60
            
            K[:,2,i,j,k][idx_K_1] = np.round((K[:,2,i,j,k][idx_K_1])        //(3.6149/39.9))
            K[:,2,i,j,k][idx_K_2] = np.round((K[:,2,i,j,k][idx_K_2] - lim_1)//(3.6149/19.9)) + 20
            K[:,2,i,j,k][idx_K_3] = np.round((K[:,2,i,j,k][idx_K_3] - lim_2)//(3.6149/9.9)) + 60
            
            O2[:,2,i,j,k][idx_O2_1] = np.round((O2[:,2,i,j,k][idx_O2_1])        //(3.6149/39.9))
            O2[:,2,i,j,k][idx_O2_2] = np.round((O2[:,2,i,j,k][idx_O2_2] - lim_1)//(3.6149/19.9)) + 20
            O2[:,2,i,j,k][idx_O2_3] = np.round((O2[:,2,i,j,k][idx_O2_3] - lim_2)//(3.6149/9.9)) + 60
            
            H2[:,2,i,j,k][idx_H2_1] = np.round((H2[:,2,i,j,k][idx_H2_1])        //(3.6149/39.9))
            H2[:,2,i,j,k][idx_H2_2] = np.round((H2[:,2,i,j,k][idx_H2_2] - lim_1)//(3.6149/19.9)) + 20
            H2[:,2,i,j,k][idx_H2_3] = np.round((H2[:,2,i,j,k][idx_H2_3] - lim_2)//(3.6149/9.9)) + 60

x_Cl = Cl[:,0,:,:,:].astype(int)
y_Cl = Cl[:,1,:,:,:].astype(int)
z_Cl = Cl[:,2,:,:,:].astype(int)

x_K = K[:,0,:,:,:].astype(int)
y_K = K[:,1,:,:,:].astype(int)
z_K = K[:,2,:,:,:].astype(int)

x_O2 = O2[:,0,:,:,:].astype(int)
y_O2 = O2[:,1,:,:,:].astype(int)
z_O2 = O2[:,2,:,:,:].astype(int)

x_H2 = H2[:,0,:,:,:].astype(int)
y_H2 = H2[:,1,:,:,:].astype(int)
z_H2 = H2[:,2,:,:,:].astype(int)

xy_bins = 160
z_bins = 96
z_bins_H2O = 20

i = 0
j = 0
k = 0
idx_Cl = np.where(z_Cl[:,i,j,k] < z_bins)
idx_K = np.where(z_K[:,i,j,k] < z_bins)
idx_O2 = np.where(z_O2[:,i,j,k] < z_bins_H2O)
idx_H2 = np.where(z_H2[:,i,j,k] < z_bins_H2O)
    
#%%    
indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i,j,k],z_K[idx_K,i,j,k],z_O2[idx_O2,i,j,k],z_H2[idx_H2,i,j,k]]),
                    np.hstack([x_Cl[idx_Cl,i,j,k],x_K[idx_K,i,j,k],x_O2[idx_O2,i,j,k],x_H2[idx_H2,i,j,k]]),
                   np.hstack([y_Cl[idx_Cl,i,j,k],y_K[idx_K,i,j,k],y_O2[idx_O2,i,j,k],y_H2[idx_H2,i,j,k]])])
    
values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))

s_train = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)
  
for i in range(1,1000):
    idx_Cl = np.where(z_Cl[:,i,j,k] < z_bins)
    idx_K = np.where(z_K[:,i,j,k] < z_bins)
    idx_O2 = np.where(z_O2[:,i,j,k] < z_bins_H2O)
    idx_H2 = np.where(z_H2[:,i,j,k] < z_bins_H2O)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i,j,k],z_K[idx_K,i,j,k],z_O2[idx_O2,i,j,k],z_H2[idx_H2,i,j,k]]),
                    np.hstack([x_Cl[idx_Cl,i,j,k],x_K[idx_K,i,j,k],x_O2[idx_O2,i,j,k],x_H2[idx_H2,i,j,k]]),
                   np.hstack([y_Cl[idx_Cl,i,j,k],y_K[idx_K,i,j,k],y_O2[idx_O2,i,j,k],y_H2[idx_H2,i,j,k]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
    
    s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

for k in range(1,4):
    for i in range(0,1000):
        idx_Cl = np.where(z_Cl[:,i,j,k] < z_bins)
        idx_K = np.where(z_K[:,i,j,k] < z_bins)
        idx_O2 = np.where(z_O2[:,i,j,k] < z_bins_H2O)
        idx_H2 = np.where(z_H2[:,i,j,k] < z_bins_H2O)
            
        indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                          np.hstack([z_Cl[idx_Cl,i,j,k],z_K[idx_K,i,j,k],z_O2[idx_O2,i,j,k],z_H2[idx_H2,i,j,k]]),
                        np.hstack([x_Cl[idx_Cl,i,j,k],x_K[idx_K,i,j,k],x_O2[idx_O2,i,j,k],x_H2[idx_H2,i,j,k]]),
                       np.hstack([y_Cl[idx_Cl,i,j,k],y_K[idx_K,i,j,k],y_O2[idx_O2,i,j,k],y_H2[idx_H2,i,j,k]])])
            
        values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
        
        s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

for k in range(4):
    for j in range(1,12):
        for i in range(0,1000):
            idx_Cl = np.where(z_Cl[:,i,j,k] < z_bins)
            idx_K = np.where(z_K[:,i,j,k] < z_bins)
            idx_O2 = np.where(z_O2[:,i,j,k] < z_bins_H2O)
            idx_H2 = np.where(z_H2[:,i,j,k] < z_bins_H2O)
                
            indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                              np.hstack([z_Cl[idx_Cl,i,j,k],z_K[idx_K,i,j,k],z_O2[idx_O2,i,j,k],z_H2[idx_H2,i,j,k]]),
                            np.hstack([x_Cl[idx_Cl,i,j,k],x_K[idx_K,i,j,k],x_O2[idx_O2,i,j,k],x_H2[idx_H2,i,j,k]]),
                           np.hstack([y_Cl[idx_Cl,i,j,k],y_K[idx_K,i,j,k],y_O2[idx_O2,i,j,k],y_H2[idx_H2,i,j,k]])])
                
            values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
            
            s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

j = 12
for k in range(4):
    for i in range(251,1000):
        idx_Cl = np.where(z_Cl[:,i,j,k] < z_bins)
        idx_K = np.where(z_K[:,i,j,k] < z_bins)
        idx_O2 = np.where(z_O2[:,i,j,k] < z_bins_H2O)
        idx_H2 = np.where(z_H2[:,i,j,k] < z_bins_H2O)
            
        indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                          np.hstack([z_Cl[idx_Cl,i,j,k],z_K[idx_K,i,j,k],z_O2[idx_O2,i,j,k],z_H2[idx_H2,i,j,k]]),
                        np.hstack([x_Cl[idx_Cl,i,j,k],x_K[idx_K,i,j,k],x_O2[idx_O2,i,j,k],x_H2[idx_H2,i,j,k]]),
                       np.hstack([y_Cl[idx_Cl,i,j,k],y_K[idx_K,i,j,k],y_O2[idx_O2,i,j,k],y_H2[idx_H2,i,j,k]])])
            
        values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
        
        s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

#%%
s_train = s_train.coalesce()

s_train.requires_grad = True

torch.save(s_train,'s_train_H2O_extended')

#%%
Cl = np.load('test_params/Cl_tensor.npy')
K = np.load('test_params/K_tensor.npy')
q_2D = np.load('test_params/q_unet_tensor_sorted.npy')
O2 = np.load('test_params/O2_tensor.npy')
H2 = np.load('test_params/H2_tensor.npy')

Cl_r = np.load('test_params/Cl_tensor_r.npy')
K_r = np.load('test_params/K_tensor_r.npy')
q_2D_r = np.load('test_params/q_tensor_r.npy')
O2_r = np.load('test_params/O2_tensor_r.npy')
H2_r = np.load('test_params/H2_tensor_r.npy')

Cl_xy = np.load('test_params/Cl_tensor_xy.npy')
K_xy = np.load('test_params/K_tensor_xy.npy')
q_2D_xy = np.load('test_params/q_tensor_xy.npy')
O2_xy = np.load('test_params/O2_tensor_xy.npy')
H2_xy = np.load('test_params/H2_tensor_xy.npy')

Cl_minxy = np.load('test_params/Cl_tensor_minxy.npy')
K_minxy = np.load('test_params/K_tensor_minxy.npy')
q_2D_minxy = np.load('test_params/q_tensor_minxy.npy')
O2_minxy = np.load('test_params/O2_tensor_minxy.npy')
H2_minxy = np.load('test_params/H2_tensor_minxy.npy')

Cl = np.stack([Cl,Cl_r,Cl_xy,Cl_minxy],axis=-1)
K = np.stack([K,K_r,K_xy,K_minxy],axis=-1)
O2 = np.stack([O2,O2_r,O2_xy,O2_minxy],axis=-1)
H2 = np.stack([H2,H2_r,H2_xy,H2_minxy],axis=-1)
q_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

Cl = Cl[:,:,:,0,:]
K = K[:,:,:,0,:]
O2 = O2[:,:,:,0,:]
H2 = H2[:,:,:,0,:]
q_2D = q_2D[:,:,:,0,:]

Cl_copy = Cl.copy()
K_copy = K.copy()
O2_copy = O2.copy()
H2_copy = H2.copy()
q_2D_copy = q_2D.copy()

width_xy = 14.459600448608398
#%%
for k in range(4):
    Cl[:,0,:,k] = np.round((Cl_copy[:,0,:,k] + width_xy)  //(3.6149/19.9))
    K[:,0,:,k] = np.round((K_copy[:,0,:,k] + width_xy)    //(3.6149/19.9))
    Cl[:,1,:,k] = np.round((Cl_copy[:,1,:,k] + width_xy)  //(3.6149/19.9))
    K[:,1,:,k] = np.round((K_copy[:,1,:,k] + width_xy)    //(3.6149/19.9))
    O2[:,0,:,k] = np.round((O2_copy[:,0,:,k] + width_xy)  //(3.6149/19.9))
    H2[:,0,:,k] = np.round((H2_copy[:,0,:,k] + width_xy)  //(3.6149/19.9))
    O2[:,1,:,k] = np.round((O2_copy[:,1,:,k] + width_xy)  //(3.6149/19.9))
    H2[:,1,:,k] = np.round((H2_copy[:,1,:,k] + width_xy)  //(3.6149/19.9))
    
    Cl[:,2,:,k] = Cl_copy[:,2,:,k] - q_2D[0,3,:,k] - H_min
    K[:,2,:,k] = K_copy[:,2,:,k] - q_2D[0,3,:,k] - H_min
    O2[:,2,:,k] = O2_copy[:,2,:,k] - q_2D[0,3,:,k] - H_min
    H2[:,2,:,k] = H2_copy[:,2,:,k] - q_2D[0,3,:,k] - H_min

#%%
lim_1 = (3.6149/39.9) * 20
lim_2 = lim_1 + (3.6149/19.9) * 40

for k in range(4):
    for i in range(500):
        idx_Cl_1 = np.where((Cl[:,2,i,k] < lim_1) & (Cl[:,2,i,k] >= 0.0))
        idx_Cl_2 = np.where((Cl[:,2,i,k] < lim_2) & (Cl[:,2,i,k] >= lim_1))
        idx_Cl_3 = np.where((Cl[:,2,i,k] >= lim_2))

        idx_K_1 = np.where((K[:,2,i,k] < lim_1) & (K[:,2,i,k] >= 0.0))
        idx_K_2 = np.where((K[:,2,i,k] < lim_2) & (K[:,2,i,k] >= lim_1)) 
        idx_K_3 = np.where((K[:,2,i,k] >= lim_2))
        
        idx_O2_1 = np.where((O2[:,2,i,k] < lim_1) & (O2[:,2,i,k] >= 0.0))
        idx_O2_2 = np.where((O2[:,2,i,k] < lim_2) & (O2[:,2,i,k] >= lim_1)) 
        idx_O2_3 = np.where((O2[:,2,i,k] >= lim_2))

        idx_H2_1 = np.where((H2[:,2,i,k] < lim_1) & (H2[:,2,i,k] >= 0.0))
        idx_H2_2 = np.where((H2[:,2,i,k] < lim_2) & (H2[:,2,i,k] >= lim_1)) 
        idx_H2_3 = np.where((H2[:,2,i,k] >= lim_2))
        
        Cl[:,2,i,k][idx_Cl_1] = np.round((Cl[:,2,i,k][idx_Cl_1])        //(3.6149/39.9))
        Cl[:,2,i,k][idx_Cl_2] = np.round((Cl[:,2,i,k][idx_Cl_2] - lim_1)//(3.6149/19.9)) + 20
        Cl[:,2,i,k][idx_Cl_3] = np.round((Cl[:,2,i,k][idx_Cl_3] - lim_2)//(3.6149/9.9)) + 60
        
        K[:,2,i,k][idx_K_1] = np.round((K[:,2,i,k][idx_K_1])        //(3.6149/39.9))
        K[:,2,i,k][idx_K_2] = np.round((K[:,2,i,k][idx_K_2] - lim_1)//(3.6149/19.9)) + 20
        K[:,2,i,k][idx_K_3] = np.round((K[:,2,i,k][idx_K_3] - lim_2)//(3.6149/9.9)) + 60
        
        O2[:,2,i,k][idx_O2_1] = np.round((O2[:,2,i,k][idx_O2_1])        //(3.6149/39.9))
        O2[:,2,i,k][idx_O2_2] = np.round((O2[:,2,i,k][idx_O2_2] - lim_1)//(3.6149/19.9)) + 20
        O2[:,2,i,k][idx_O2_3] = np.round((O2[:,2,i,k][idx_O2_3] - lim_2)//(3.6149/9.9)) + 60
        
        H2[:,2,i,k][idx_H2_1] = np.round((H2[:,2,i,k][idx_H2_1])        //(3.6149/39.9))
        H2[:,2,i,k][idx_H2_2] = np.round((H2[:,2,i,k][idx_H2_2] - lim_1)//(3.6149/19.9)) + 20
        H2[:,2,i,k][idx_H2_3] = np.round((H2[:,2,i,k][idx_H2_3] - lim_2)//(3.6149/9.9)) + 60

x_Cl = Cl[:,0,:,:].astype(int)
y_Cl = Cl[:,1,:,:].astype(int)
z_Cl = Cl[:,2,:,:].astype(int)

x_K = K[:,0,:,:].astype(int)
y_K = K[:,1,:,:].astype(int)
z_K = K[:,2,:,:].astype(int)

x_O2 = O2[:,0,:,:].astype(int)
y_O2 = O2[:,1,:,:].astype(int)
z_O2 = O2[:,2,:,:].astype(int)

x_H2 = H2[:,0,:,:].astype(int)
y_H2 = H2[:,1,:,:].astype(int)
z_H2 = H2[:,2,:,:].astype(int)

i = 0
k = 0
idx_Cl = np.where(z_Cl[:,i,k] < z_bins)
idx_K = np.where(z_K[:,i,k] < z_bins)
idx_O2 = np.where(z_O2[:,i,k] < z_bins_H2O)
idx_H2 = np.where(z_H2[:,i,k] < z_bins_H2O)
    
indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i,k],z_K[idx_K,i,k],z_O2[idx_O2,i,k],z_H2[idx_H2,i,k]]),
                    np.hstack([x_Cl[idx_Cl,i,k],x_K[idx_K,i,k],x_O2[idx_O2,i,k],x_H2[idx_H2,i,k]]),
                   np.hstack([y_Cl[idx_Cl,i,k],y_K[idx_K,i,k],y_O2[idx_O2,i,k],y_H2[idx_H2,i,k]])])

values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))

s_test = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)
for i in range(1,500):
    idx_Cl = np.where(z_Cl[:,i,k] < z_bins)
    idx_K = np.where(z_K[:,i,k] < z_bins)
    idx_O2 = np.where(z_O2[:,i,k] < z_bins_H2O)
    idx_H2 = np.where(z_H2[:,i,k] < z_bins_H2O)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                      np.hstack([z_Cl[idx_Cl,i,k],z_K[idx_K,i,k],z_O2[idx_O2,i,k],z_H2[idx_H2,i,k]]),
                    np.hstack([x_Cl[idx_Cl,i,k],x_K[idx_K,i,k],x_O2[idx_O2,i,k],x_H2[idx_H2,i,k]]),
                   np.hstack([y_Cl[idx_Cl,i,k],y_K[idx_K,i,k],y_O2[idx_O2,i,k],y_H2[idx_H2,i,k]])])
    
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
    
    s_test = torch.cat([s_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

for k in range(1,4):
    for i in range(0,500):
        idx_Cl = np.where(z_Cl[:,i,k] < z_bins)
        idx_K = np.where(z_K[:,i,k] < z_bins)
        idx_O2 = np.where(z_O2[:,i,k] < z_bins_H2O)
        idx_H2 = np.where(z_H2[:,i,k] < z_bins_H2O)
            
        indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1]+np.shape(idx_O2)[1]+np.shape(idx_H2)[1])),
                          np.hstack([z_Cl[idx_Cl,i,k],z_K[idx_K,i,k],z_O2[idx_O2,i,k],z_H2[idx_H2,i,k]]),
                        np.hstack([x_Cl[idx_Cl,i,k],x_K[idx_K,i,k],x_O2[idx_O2,i,k],x_H2[idx_H2,i,k]]),
                       np.hstack([y_Cl[idx_Cl,i,k],y_K[idx_K,i,k],y_O2[idx_O2,i,k],y_H2[idx_H2,i,k]])])
        
        values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1]),-0.8476*np.ones(np.shape(idx_O2)[1]),0.4238*np.ones(np.shape(idx_H2)[1])]))
        
        s_test = torch.cat([s_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

#%%
s_test = s_test.coalesce()
torch.save(s_test,'s_test_H2O_extended')

device = torch.device('cpu')

#%%
q_train = np.load('train_params/q_tensor.npy')
q_test = np.load('test_params/q_tensor.npy')

q_2D = np.load('train_params/q_unet_tensor_sorted.npy')
q_2D_r = np.load('train_params/q_tensor_r.npy')
q_2D_xy = np.load('train_params/q_tensor_xy.npy')
q_2D_minxy = np.load('train_params/q_tensor_minxy.npy')

q_train_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

q_2D = np.load('test_params/q_unet_tensor_sorted.npy')
q_2D_r = np.load('test_params/q_tensor_r.npy')
q_2D_xy = np.load('test_params/q_tensor_xy.npy')
q_2D_minxy = np.load('test_params/q_tensor_minxy.npy')

q_test_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

q_train_2D = q_train_2D[:,0,:,:]
q_test_2D = q_test_2D[:,0,:,:]

#%%

q_train_copy = q_train.copy()
q_test_copy = q_test.copy()
q_train = np.transpose(q_train_copy)
q_test = np.transpose(q_test_copy)

tensor_y_train = torch.Tensor(q_train)
tensor_y_test = torch.Tensor(q_test)

q_train_2D_copy = q_train_2D.copy()
q_test_2D_copy = q_test_2D.copy()
q_train_2D = np.transpose(q_train_2D_copy)
q_test_2D = np.transpose(q_test_2D_copy)

tensor_y_train_2D = torch.Tensor(q_train_2D)
tensor_y_test_2D = torch.Tensor(q_test_2D)

#%%
tensor_y_train = torch.Tensor(q_train)
tensor_y_test = torch.Tensor(q_test)
train_y = tensor_y_train[0,:,:]
for i in range(1,12):
    train_y = torch.vstack([train_y, tensor_y_train[i,:,:]])
train_y = torch.vstack([train_y, tensor_y_train[12,251:,:]])   
test_y = tensor_y_test[0,:,:]

k = 0
train_2D_y = tensor_y_train_2D[k,0,:,:]
for k in range(1,4):
    train_2D_y = torch.vstack([train_2D_y, tensor_y_train_2D[k,0,:,:]])
for k in range(4):
    for i in range(1,12):
        train_2D_y = torch.vstack([train_2D_y, tensor_y_train_2D[k,i,:,:]])
for k in range(4):
    train_2D_y = torch.vstack([train_2D_y, tensor_y_train_2D[k,12,251:,:]])   
    
#%%
test_2D_y = tensor_y_test_2D[0,0,:,:]
for k in range(1,4):
    test_2D_y = torch.vstack([test_2D_y, tensor_y_test_2D[k,0,:,:]])
    
#%%
s_train_3D = s_train.to(device=device, dtype=torch.float)

train_data = TensorDataset(s_train_3D,train_2D_y) 
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
torch.save(train_loader,'train_loader_H2O_extended_3D')

#%%
s_test_3D = s_test.to(device=device, dtype=torch.float)

test_data = TensorDataset(s_test_3D,test_2D_y) 
test_loader = DataLoader(test_data, batch_size=1)
torch.save(test_loader,'test_loader_H2O_extended_3D')
