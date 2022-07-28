import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

#%%
X = np.load('train_params/X_tensor.npy')
X_H2O = np.load('train_params/X_H2O_tensor.npy')
q_2D = np.load('train_params/q_unet_tensor_sorted.npy')
K = np.load('train_params/K_tensor.npy')
H2 = np.load('train_params/H2_tensor.npy')

X_r = np.load('train_params/X_tensor_r.npy')
X_H2O_r = np.load('train_params/X_H2O_tensor_r.npy')
K_r = np.load('train_params/K_tensor_r.npy')
q_2D_r = np.load('train_params/q_tensor_r.npy')

X_xy = np.load('train_params/X_tensor_xy.npy')
X_H2O_xy = np.load('train_params/X_H2O_tensor_xy.npy')
K_xy = np.load('train_params/K_tensor_xy.npy')
q_2D_xy = np.load('train_params/q_tensor_xy.npy')

X_minxy = np.load('train_params/X_tensor_minxy.npy')
X_H2O_minxy = np.load('train_params/X_H2O_tensor_minxy.npy')
K_minxy = np.load('train_params/K_tensor_minxy.npy')
q_2D_minxy = np.load('train_params/q_tensor_minxy.npy')

X = np.stack([X,X_r,X_xy,X_minxy],axis=-1)
X_H2O = np.stack([X_H2O,X_H2O_r,X_H2O_xy,X_H2O_minxy],axis=-1)
K = np.stack([K,K_r,K_xy,K_minxy],axis=-1)
q_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

#%%
K_copy = K.copy()
width_xy = 14.459600448608398

H2_copy = H2.copy()
X_copy = X.copy()
X_H2O_copy = X_H2O.copy()
q_2D_copy = q_2D.copy()

#%%
for k in range(4):
    for j in range(13):
        X[:,2,:,j,k] = X_copy[:,2,:,j,k] - q_2D_copy[0,3,:,j,k] 
        X_H2O[:,2,:,j,k] = X_H2O_copy[:,2,:,j,k] - q_2D_copy[0,3,:,j,k]
        K[:,2,:,j,k] = K_copy[:,2,:,j,k] - q_2D_copy[0,3,:,j,k] 
for j in range(13):   
    H2[:,2,:,j] = H2[:,2,:,j] - q_2D_copy[0,3,:,j,0]

H_min = np.min(H2[:,2,:,:12])
K_min = np.min(K[:,2,:,:12,:])
X[:,2,:,:,:] = X[:,2,:,:,:] - K_min
X_H2O[:,2,:,:,:] = X_H2O[:,2,:,:,:] - H_min

#%%
X = np.transpose(X.reshape((37+37)*3,1000,13,4))
X_H2O = np.transpose(X_H2O.reshape((37+37+50*3)*3,1000,13,4))

#%%
X_tensor = torch.Tensor(X)                  
X_tensor.requires_grad = True 
X_H2O_tensor = torch.Tensor(X_H2O)                  
X_H2O_tensor.requires_grad = True    

X_tensor_copy = torch.clone(X_tensor)
X_H2O_tensor_copy = torch.clone(X_H2O_tensor)

k=0
X_tensor = X_tensor_copy[k,0,:,:]
for k in range(1,4):
    X_tensor = torch.vstack([X_tensor, X_tensor_copy[k,0,:,:]])
for k in range(4):
    for i in range(1,12):
        X_tensor = torch.vstack([X_tensor, X_tensor_copy[k,i,:,:]])
for k in range(4):
    X_tensor = torch.vstack([X_tensor, X_tensor_copy[k,12,251:,:]])

k=0
X_H2O_tensor = X_H2O_tensor_copy[k,0,:,:]
for k in range(1,4):
    X_H2O_tensor = torch.vstack([X_H2O_tensor, X_H2O_tensor_copy[k,0,:,:]])
for k in range(4):
    for i in range(1,12):
        X_H2O_tensor = torch.vstack([X_H2O_tensor, X_H2O_tensor_copy[k,i,:,:]])
for k in range(4):
    X_H2O_tensor = torch.vstack([X_H2O_tensor, X_H2O_tensor_copy[k,12,251:,:]])

train_X_tensor = X_tensor
train_X_H2O_tensor = X_H2O_tensor

#%%
torch.save(train_X_tensor,'X_tensor_train_extended')
torch.save(train_X_H2O_tensor,'X_H2O_tensor_train_extended')

#%%
device = torch.device('cpu')
#train_X_tensor = torch.load('X_tensor_train_extended')

#%% test
X = np.load('test_params/X_tensor.npy')
X_H2O = np.load('test_params/X_H2O_tensor.npy')
q_2D = np.load('test_params/q_unet_tensor_sorted.npy')

X_r = np.load('test_params/X_tensor_r.npy')
X_H2O_r = np.load('test_params/X_H2O_tensor_r.npy')
q_2D_r = np.load('test_params/q_tensor_r.npy')

X_xy = np.load('test_params/X_tensor_xy.npy')
X_H2O_xy = np.load('test_params/X_H2O_tensor_xy.npy')
q_2D_xy = np.load('test_params/q_tensor_xy.npy')

X_minxy = np.load('test_params/X_tensor_minxy.npy')
X_H2O_minxy = np.load('test_params/X_H2O_tensor_minxy.npy')
q_2D_minxy = np.load('test_params/q_tensor_minxy.npy')

X = np.stack([X,X_r,X_xy,X_minxy],axis=-1)
X_H2O = np.stack([X_H2O,X_H2O_r,X_H2O_xy,X_H2O_minxy],axis=-1)
q_2D = np.stack([q_2D,q_2D_r,q_2D_xy,q_2D_minxy],axis=-1)

#%%
X = X[:,:,:,0,:]
X_H2O = X_H2O[:,:,:,0,:]
q_2D = q_2D[:,:,:,0,:]

X_copy = X.copy()
X_H2O_copy = X_H2O.copy()

#%%
for k in range(4):
    X[:,2,:,k] = X_copy[:,2,:,k] - q_2D[0,3,:,k] - K_min
    X_H2O[:,2,:,k] = X_H2O_copy[:,2,:,k] - q_2D[0,3,:,k] - H_min

X = np.transpose(X.reshape((37+37)*3,500,4))
X_H2O = np.transpose(X_H2O.reshape((37+37+50*3)*3,500,4))

#%%
X_tensor = torch.Tensor(X)     
X_H2O_tensor = torch.Tensor(X_H2O)             

X_tensor_copy = torch.clone(X_tensor)
X_H2O_tensor_copy = torch.clone(X_H2O_tensor)

k=0
X_tensor = X_tensor_copy[k,:,:]
for k in range(1,4):
    X_tensor = torch.vstack([X_tensor, X_tensor_copy[k,:,:]])

k=0
X_H2O_tensor = X_H2O_tensor_copy[k,:,:]
for k in range(1,4):
    X_H2O_tensor = torch.vstack([X_H2O_tensor, X_H2O_tensor_copy[k,:,:]])

test_X_tensor = X_tensor
test_X_H2O_tensor = X_H2O_tensor

#%%
torch.save(test_X_tensor,'X_tensor_test_extended')
torch.save(test_X_H2O_tensor,'X_H2O_tensor_test_extended')
#%%
device = torch.device('cpu')
#train_X_tensor = torch.load('X_tensor_train')
#test_X_tensor = torch.load('X_tensor_test')

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
train_data = TensorDataset(train_X_tensor,train_2D_y)
train_data_H2O = TensorDataset(train_X_H2O_tensor,train_2D_y)

test_data = TensorDataset(test_X_tensor,test_2D_y)
test_data_H2O = TensorDataset(test_X_H2O_tensor,test_2D_y)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100)

train_loader_H2O = DataLoader(train_data_H2O, batch_size=8, shuffle=True)
test_loader_H2O = DataLoader(test_data_H2O, batch_size=100)

#%%
torch.save(train_loader,'train_loader_X_extended')
torch.save(test_loader,'test_loader_X_extended')
torch.save(train_loader_H2O,'train_loader_X_H2O_extended')
torch.save(test_loader_H2O,'test_loader_X_H2O_extended')
