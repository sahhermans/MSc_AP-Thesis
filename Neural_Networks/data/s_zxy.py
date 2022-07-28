import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

#%%
Cl = np.load('train_params/Cl_tensor.npy')
K = np.load('train_params/K_tensor.npy')
q_2D = np.load('train_params/q_unet_tensor_sorted.npy')

Cl_r = np.load('train_params/Cl_tensor_r.npy')
K_r = np.load('train_params/K_tensor_r.npy')
q_2D_r = np.load('train_params/q_tensor_r.npy')

Cl_xy = np.load('train_params/Cl_tensor_xy.npy')
K_xy = np.load('train_params/K_tensor_xy.npy')
q_2D_xy = np.load('train_params/q_tensor_xy.npy')

Cl_minxy = np.load('train_params/Cl_tensor_minxy.npy')
K_minxy = np.load('train_params/K_tensor_minxy.npy')
q_2D_minxy = np.load('train_params/q_tensor_minxy.npy')

#%%
Cl = np.stack([Cl,Cl_r,Cl_xy,Cl_minxy],axis=3)
K = np.stack([K,K_r,K_xy,K_minxy],axis=3)
q_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=3)

Cl_copy = Cl.copy()
K_copy = K.copy()
q_2D_copy = q_2D.copy()

#%%   
width_xy = 14.459600448608398
for j in range(13*4):
        Cl[:,0,:,j] = np.round((Cl_copy[:,0,:,j] + width_xy)  //(3.6149/19.9))
        K[:,0,:,j] = np.round((K_copy[:,0,:,j] + width_xy)    //(3.6149/19.9))
        Cl[:,1,:,j] = np.round((Cl_copy[:,1,:,j] + width_xy)  //(3.6149/19.9))
        K[:,1,:,j] = np.round((K_copy[:,1,:,j] + width_xy)    //(3.6149/19.9))
        
        Cl[:,2,:,j] = Cl_copy[:,2,:,j] - q_2D[0,3,:,j] 
        K[:,2,:,j] = K_copy[:,2,:,j] - q_2D[0,3,:,j] 

K_min = np.min(K[:,2,:,:12])
Cl[:,2,:,:] = Cl[:,2,:,:] - K_min
K[:,2,:,:] = K[:,2,:,:] - K_min
    
lim_1 = (3.6149/39.9) * 10
lim_2 = lim_1 + (3.6149/19.9) * 40

for k in range(4):
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
k = 0
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

for j in range(13,25):
    for i in range(0,1000):
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
for k in range(4):
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
s_train = s_train.coalesce()
s_train.requires_grad = True
torch.save(s_train,'s_train_d_extended')

#%%
Cl = np.load('test_params/Cl_tensor.npy')
K = np.load('test_params/K_tensor.npy')
q_2D = np.load('test_params/q_unet_tensor_sorted.npy')

Cl_r = np.load('test_params/Cl_tensor_r.npy')
K_r = np.load('test_params/K_tensor_r.npy')
q_2D_r = np.load('test_params/q_tensor_r.npy')

Cl_xy = np.load('test_params/Cl_tensor_xy.npy')
K_xy = np.load('test_params/K_tensor_xy.npy')
q_2D_xy = np.load('test_params/q_tensor_xy.npy')

Cl_minxy = np.load('test_params/Cl_tensor_minxy.npy')
K_minxy = np.load('test_params/K_tensor_minxy.npy')
q_2D_minxy = np.load('test_params/q_tensor_minxy.npy')

Cl = np.stack([Cl,Cl_r,Cl_xy,Cl_minxy],axis=-1)
K = np.stack([K,K_r,K_xy,K_minxy],axis=-1)
q_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

Cl = Cl[:,:,:,0,:]
K = K[:,:,:,0,:]
q_2D = q_2D[:,:,:,0,:]

Cl_copy = Cl.copy()
K_copy = K.copy()
q_2D_copy = q_2D.copy()

width_xy = 14.459600448608398

for k in range(4):
    Cl[:,0,:,k] = np.round((Cl_copy[:,0,:,k] + width_xy)  //(3.6149/19.9))
    K[:,0,:,k] = np.round((K_copy[:,0,:,k] + width_xy)    //(3.6149/19.9))
    Cl[:,1,:,k] = np.round((Cl_copy[:,1,:,k] + width_xy)  //(3.6149/19.9))
    K[:,1,:,k] = np.round((K_copy[:,1,:,k] + width_xy)    //(3.6149/19.9))
    
    Cl[:,2,:,k] = Cl_copy[:,2,:,k] - q_2D[0,3,:,k] - K_min
    K[:,2,:,k] = K_copy[:,2,:,k] - q_2D[0,3,:,k] - K_min

lim_1 = (3.6149/39.9) * 10
lim_2 = lim_1 + (3.6149/19.9) * 40

for k in range(4):
    for i in range(500):
        idx_Cl_1 = np.where((Cl[:,2,i,k] < lim_1) & (Cl[:,2,i,k] >= 0.0))
        idx_Cl_2 = np.where((Cl[:,2,i,k] < lim_2) & (Cl[:,2,i,k] >= lim_1))
        idx_Cl_3 = np.where((Cl[:,2,i,k] >= lim_2))
    
        idx_K_1 = np.where((K[:,2,i,k] < lim_1) & (K[:,2,i,k] >= 0.0))
        idx_K_2 = np.where((K[:,2,i,k] < lim_2) & (K[:,2,i,k] >= lim_1)) 
        idx_K_3 = np.where((K[:,2,i,k] >= lim_2))
        
        Cl[:,2,i,k][idx_Cl_1] = np.round((Cl[:,2,i,k][idx_Cl_1])        //(3.6149/39.9))
        Cl[:,2,i,k][idx_Cl_2] = np.round((Cl[:,2,i,k][idx_Cl_2] - lim_1)//(3.6149/19.9)) + 10
        Cl[:,2,i,k][idx_Cl_3] = np.round((Cl[:,2,i,k][idx_Cl_3] - lim_2)//(3.6149/9.9)) + 50
        
        K[:,2,i,k][idx_K_1] = np.round((K[:,2,i,k][idx_K_1])        //(3.6149/39.9))
        K[:,2,i,k][idx_K_2] = np.round((K[:,2,i,k][idx_K_2] - lim_1)//(3.6149/19.9)) + 10
        K[:,2,i,k][idx_K_3] = np.round((K[:,2,i,k][idx_K_3] - lim_2)//(3.6149/9.9)) + 50

x_Cl = Cl[:,0,:,:].astype(int)
y_Cl = Cl[:,1,:,:].astype(int)
z_Cl = Cl[:,2,:,:].astype(int)

x_K = K[:,0,:,:].astype(int)
y_K = K[:,1,:,:].astype(int)
z_K = K[:,2,:,:].astype(int)

i = 0
k = 0
idx_Cl = np.where(z_Cl[:,i,k] < z_bins)
idx_K = np.where(z_K[:,i,k] < z_bins)

indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])), 
                      np.hstack([z_Cl[idx_Cl,i,k],z_K[idx_K,i,k]]),
                    np.hstack([x_Cl[idx_Cl,i,k],x_K[idx_K,i,k]]),
                   np.hstack([y_Cl[idx_Cl,i,k],y_K[idx_K,i,k]])])
    
values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
#%%
s_test = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)

for i in range(1,500):
    idx_Cl = np.where(z_Cl[:,i,k] < z_bins)
    idx_K = np.where(z_K[:,i,k] < z_bins)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                          np.hstack([z_Cl[idx_Cl,i,k],z_K[idx_K,i,k]]),
                        np.hstack([x_Cl[idx_Cl,i,k],x_K[idx_K,i,k]]),
                       np.hstack([y_Cl[idx_Cl,i,k],y_K[idx_K,i,k]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
        
    s_test = torch.cat([s_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

for k in range(1,4):
    for i in range(0,500):
        idx_Cl = np.where(z_Cl[:,i,k] < z_bins)
        idx_K = np.where(z_K[:,i,k] < z_bins)
            
        indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                              np.hstack([z_Cl[idx_Cl,i,k],z_K[idx_K,i,k]]),
                            np.hstack([x_Cl[idx_Cl,i,k],x_K[idx_K,i,k]]),
                           np.hstack([y_Cl[idx_Cl,i,k],y_K[idx_K,i,k]])])
            
        values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
            
        s_test = torch.cat([s_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, z_bins, xy_bins, xy_bins), dtype=torch.float64), dim=0)], dim=0)

#%%
s_test = s_test.coalesce()
torch.save(s_test,'s_test_d_extended')

device = torch.device('cpu')
#%%
s_train_d = torch.load('s_train_d_extended')
s_test_d = torch.load('s_test_d_extended')

#%%
q_train = np.load('train_params/q_tensor.npy')

q_2D = np.load('train_params/q_unet_tensor_sorted.npy')
q_2D_r = np.load('train_params/q_tensor_r.npy')
q_2D_xy = np.load('train_params/q_tensor_xy.npy')
q_2D_minxy = np.load('train_params/q_tensor_minxy.npy')

q_train_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

q_test = np.load('test_params/q_tensor.npy')

q_2D = np.load('test_params/q_unet_tensor_sorted.npy')
q_2D_r = np.load('test_params/q_tensor_r.npy')
q_2D_xy = np.load('test_params/q_tensor_xy.npy')
q_2D_minxy = np.load('test_params/q_tensor_minxy.npy')

q_test_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

#%%
q_train_2D = q_train_2D[:,0,:,:,:]
q_test_2D = q_test_2D[:,0,:,:,:]

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
torch.save(train_loader,'train_loader_d_extended')

#%%
s_test_3D = s_test.to(device=device, dtype=torch.float)

test_data = TensorDataset(s_test_3D,test_2D_y) 
test_loader = DataLoader(test_data, batch_size=4)
torch.save(test_loader,'test_loader_d_extended')
#%%
"""
#%%
data = [
    [0, 0, 2.1, 0, 0],
    [0, 1, 1.4, 3, 0],
    [0, 0, 4.0, 0, 0]
]

def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                coords.append([i, j])
                feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)

#coords0, feats0 = to_sparse_coo(data)

#coords1, feats1 = ME.utils.sparse_collate(coordinates=[coords0], features=[feats0])
"""