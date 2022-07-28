import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

#%%
Cl = np.load('train_params/Cl_tensor.npy')
K = np.load('train_params/K_tensor.npy')
q_2D = np.load('train_params/q_unet_tensor.npy')

Cl_copy = Cl.copy()
K_copy = K.copy()
q_2D_copy = q_2D.copy()

#%%   
width_xy = 14.459600448608398
for j in range(13):
    Cl[:,0,:,j] = np.round((Cl_copy[:,0,:,j] + width_xy)  //(3.6149/19.9))
    K[:,0,:,j] = np.round((K_copy[:,0,:,j] + width_xy)    //(3.6149/19.9))
    Cl[:,1,:,j] = np.round((Cl_copy[:,1,:,j] + width_xy)  //(3.6149/19.9))
    K[:,1,:,j] = np.round((K_copy[:,1,:,j] + width_xy)    //(3.6149/19.9))
    
    Cl[:,2,:,j] = Cl_copy[:,2,:,j] - q_2D[0,3,:,j] 
    K[:,2,:,j] = K_copy[:,2,:,j] - q_2D[0,3,:,j] 

K_min = np.min(K[:,2,:,:12])
Cl[:,2,:,:] = Cl[:,2,:,:] - K_min
K[:,2,:,:] = K[:,2,:,:] - K_min
#%%
lim_1 = (3.6149/39.9) * 10
lim_2 = lim_1 + (3.6149/19.9) * 40
    
for j in range(13):
    for i in range(1000):
        idx_Cl_1 = np.where((Cl[:,2,i,j] < lim_1) & (Cl[:,2,i,j] >= 0.0))
        idx_Cl_2 = np.where((Cl[:,2,i,j] < lim_2) & (Cl[:,2,i,j] >= lim_1))
        idx_Cl_3 = np.where((Cl[:,2,i,j] >= lim_2))
    
        idx_K_1 = np.where((K[:,2,i,j] < lim_1) & (K[:,2,i,j] >= 0.0))
        idx_K_2 = np.where((K[:,2,i,j] < lim_2) & (K[:,2,i,j] >= lim_1)) 
        idx_K_3 = np.where((K[:,2,i,j] >= lim_2))
        
        Cl[:,2,i,j][idx_Cl_1] = np.round((Cl[:,2,i,j][idx_Cl_1])        //(3.6149/39.9))
        Cl[:,2,i,j][idx_Cl_2] = np.round((Cl[:,2,i,j][idx_Cl_2] - lim_1)//(3.6149/19.9)) + 10
        Cl[:,2,i,j][idx_Cl_3] = np.round((Cl[:,2,i,j][idx_Cl_3] - lim_2)//(3.6149/9.9)) + 50
        
        K[:,2,i,j][idx_K_1] = np.round((K[:,2,i,j][idx_K_1])        //(3.6149/39.9))
        K[:,2,i,j][idx_K_2] = np.round((K[:,2,i,j][idx_K_2] - lim_1)//(3.6149/19.9)) + 10
        K[:,2,i,j][idx_K_3] = np.round((K[:,2,i,j][idx_K_3] - lim_2)//(3.6149/9.9)) + 50

x_Cl = Cl[:,0,:,:].astype(int)
y_Cl = Cl[:,1,:,:].astype(int)
z_Cl = Cl[:,2,:,:].astype(int)

x_K = K[:,0,:,:].astype(int)
y_K = K[:,1,:,:].astype(int)
z_K = K[:,2,:,:].astype(int)

xy_bins = 160
z_bins = 96

i = 0
j = 0
idx_Cl = np.where(z_Cl[:,i,j] < z_bins)
idx_K = np.where(z_K[:,i,j] < z_bins)

indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                      np.hstack([z_Cl[idx_Cl,i,j],z_K[idx_K,i,j]]),
                    np.hstack([x_Cl[idx_Cl,i,j],x_K[idx_K,i,j]]),
                   np.hstack([y_Cl[idx_Cl,i,j],y_K[idx_K,i,j]])])
   
values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
#%% 
s_train = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)
for i in range(1,1000):
    idx_Cl = np.where(z_Cl[:,i,j] < z_bins)
    idx_K = np.where(z_K[:,i,j] < z_bins)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                          np.hstack([z_Cl[idx_Cl,i,j],z_K[idx_K,i,j]]),
                        np.hstack([x_Cl[idx_Cl,i,j],x_K[idx_K,i,j]]),
                       np.hstack([y_Cl[idx_Cl,i,j],y_K[idx_K,i,j]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
        
    s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

for j in range(1,12):
    for i in range(0,1000):
        idx_Cl = np.where(z_Cl[:,i,j] < z_bins)
        idx_K = np.where(z_K[:,i,j] < z_bins)
            
        indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                              np.hstack([z_Cl[idx_Cl,i,j],z_K[idx_K,i,j]]),
                            np.hstack([x_Cl[idx_Cl,i,j],x_K[idx_K,i,j]]),
                           np.hstack([y_Cl[idx_Cl,i,j],y_K[idx_K,i,j]])])
            
        values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
            
        s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)
j = 12
for i in range(251,1000):
    idx_Cl = np.where(z_Cl[:,i,j] < z_bins)
    idx_K = np.where(z_K[:,i,j] < z_bins)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                          np.hstack([z_Cl[idx_Cl,i,j],z_K[idx_K,i,j]]),
                        np.hstack([x_Cl[idx_Cl,i,j],x_K[idx_K,i,j]]),
                       np.hstack([y_Cl[idx_Cl,i,j],y_K[idx_K,i,j]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
        
    s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

#%%
Cl = np.load('test_params/Cl_tensor.npy')
K = np.load('test_params/K_tensor.npy')
q_2D = np.load('test_params/q_unet_tensor.npy')

Cl = Cl[:,:,:,0]
K = K[:,:,:,0]
q_2D = q_2D[:,:,:,0]

Cl_copy = Cl.copy()
K_copy = K.copy()
q_2D_copy = q_2D.copy()

width_xy = 14.459600448608398

Cl[:,0,:] = np.round((Cl_copy[:,0,:] + width_xy)  //(3.6149/19.9))
K[:,0,:] = np.round((K_copy[:,0,:] + width_xy)    //(3.6149/19.9))
Cl[:,1,:] = np.round((Cl_copy[:,1,:] + width_xy)  //(3.6149/19.9))
K[:,1,:] = np.round((K_copy[:,1,:] + width_xy)    //(3.6149/19.9))
#%%
Cl[:,2,:] = Cl_copy[:,2,:] - q_2D[0,3,:] - K_min
K[:,2,:] = K_copy[:,2,:] - q_2D[0,3,:] - K_min

lim_1 = (3.6149/39.9) * 10
lim_2 = lim_1 + (3.6149/19.9) * 40

for i in range(500):
    idx_Cl_1 = np.where((Cl[:,2,i] < lim_1) & (Cl[:,2,i] >= 0.0))
    idx_Cl_2 = np.where((Cl[:,2,i] < lim_2) & (Cl[:,2,i] >= lim_1))
    idx_Cl_3 = np.where((Cl[:,2,i] >= lim_2))

    idx_K_1 = np.where((K[:,2,i] < lim_1) & (K[:,2,i] >= 0.0))
    idx_K_2 = np.where((K[:,2,i] < lim_2) & (K[:,2,i] >= lim_1)) 
    idx_K_3 = np.where((K[:,2,i] >= lim_2))
    
    Cl[:,2,i][idx_Cl_1] = np.round((Cl[:,2,i][idx_Cl_1])        //(3.6149/39.9))
    Cl[:,2,i][idx_Cl_2] = np.round((Cl[:,2,i][idx_Cl_2] - lim_1)//(3.6149/19.9)) + 10
    Cl[:,2,i][idx_Cl_3] = np.round((Cl[:,2,i][idx_Cl_3] - lim_2)//(3.6149/9.9)) + 50
    
    K[:,2,i][idx_K_1] = np.round((K[:,2,i][idx_K_1])        //(3.6149/39.9))
    K[:,2,i][idx_K_2] = np.round((K[:,2,i][idx_K_2] - lim_1)//(3.6149/19.9)) + 10
    K[:,2,i][idx_K_3] = np.round((K[:,2,i][idx_K_3] - lim_2)//(3.6149/9.9)) + 50

x_Cl = Cl[:,0,:].astype(int)
y_Cl = Cl[:,1,:].astype(int)
z_Cl = Cl[:,2,:].astype(int)

x_K = K[:,0,:].astype(int)
y_K = K[:,1,:].astype(int)
z_K = K[:,2,:].astype(int)

i = 0
idx_Cl = np.where(z_Cl[:,i] < z_bins)
idx_K = np.where(z_K[:,i] < z_bins)

indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])), 
                      np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i]]),
                    np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i]]),
                   np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i]])])
    
values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
#%%
s_test = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)

for i in range(1,500):
    idx_Cl = np.where(z_Cl[:,i] < z_bins)
    idx_K = np.where(z_K[:,i] < z_bins)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                          np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i]]),
                        np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i]]),
                       np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
        
    s_test = torch.cat([s_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

#%%
s_train = s_train.coalesce()
s_test = s_test.coalesce()

s_train.requires_grad = True

torch.save(s_train,'s_train_d')
torch.save(s_test,'s_test_d')

device = torch.device('cpu')
#%%
s_train_d = torch.load('s_train_d')
s_test_d = torch.load('s_test_d')

#%%
q_train = np.load('train_params/q_tensor.npy')
q_test = np.load('test_params/q_tensor.npy')
q_train_copy = q_train.copy()
q_test_copy = q_test.copy()
q_train = np.transpose(q_train_copy)
q_test = np.transpose(q_test_copy)

tensor_y_train = torch.Tensor(q_train)
tensor_y_test = torch.Tensor(q_test)
train_y = tensor_y_train[0,:,:]
for i in range(1,12):
    train_y = torch.vstack([train_y, tensor_y_train[i,:,:]])
train_y = torch.vstack([train_y, tensor_y_train[12,251:,:]])   
test_y = tensor_y_test[0,:,:]

#%%
s_train_3D = s_train.to(device=device, dtype=torch.float)

train_data = TensorDataset(s_train_3D,train_y) 
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
torch.save(train_loader,'train_loader_d_3D')

#%%
s_test_3D = s_test.to(device=device, dtype=torch.float)

test_data = TensorDataset(s_test_3D,test_y) 
test_loader = DataLoader(test_data, batch_size=4)
torch.save(test_loader,'test_loader_d_3D')
