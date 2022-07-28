import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

#%%
X = np.load('train_params/X_tensor.npy')
X_q = np.load('train_params/X_q_tensor.npy')
X_H2O = np.load('train_params/X_H2O_tensor.npy')
q_2D = np.load('train_params/q_unet_tensor.npy')
K = np.load('train_params/K_tensor.npy')
H2 = np.load('train_params/H2_tensor.npy')

#%%
K_copy = K.copy()
width_xy = 14.459600448608398

H2_copy = H2.copy()
X_copy = X.copy()
X_H2O_copy = X_H2O.copy()
X_q_copy = X_q.copy()
q_2D_copy = q_2D.copy()

#%%
for j in range(13):
    X[:,2,:,j] = X_copy[:,2,:,j] - q_2D_copy[0,3,:,j] 
    X_H2O[:,2,:,j] = X_H2O_copy[:,2,:,j] - q_2D_copy[0,3,:,j]
    X_q[:,3,:,j] = X_q_copy[:,3,:,j] - q_2D_copy[0,3,:,j] 
    K[:,2,:,j] = K_copy[:,2,:,j] - q_2D_copy[0,3,:,j] 
    H2[:,2,:,j] = H2_copy[:,2,:,j] - q_2D_copy[0,3,:,j]

H_min = np.min(H2[:,2,:,:12])
K_min = np.min(K[:,2,:,:12])
X[:,2,:,:] = X[:,2,:,:] - K_min
X_H2O[:,2,:,:] = X_H2O[:,2,:,:] - H_min
X_q[:,3,:,:] =  X_q[:,3,:,:] - K_min

#%%
X = np.transpose(X.reshape((37+37)*3,1000,13))
X_H2O = np.transpose(X_H2O.reshape((37+37+50*3)*3,1000,13))
X_q = np.transpose(X_q.reshape((37+37)*4,1000,13))

#%%
X_tensor = torch.Tensor(X)                  
X_tensor.requires_grad = True 
X_H2O_tensor = torch.Tensor(X_H2O)                  
X_H2O_tensor.requires_grad = True 
X_q_tensor = torch.Tensor(X_q)                  
X_q_tensor.requires_grad = True 

X_tensor_copy = torch.clone(X_tensor)
X_H2O_tensor_copy = torch.clone(X_H2O_tensor)
X_q_tensor_copy = torch.clone(X_q_tensor)

X_tensor = X_tensor_copy[0,:,:]
for i in range(1,12):
    X_tensor = torch.vstack([X_tensor, X_tensor_copy[i,:,:]])
X_tensor = torch.vstack([X_tensor, X_tensor_copy[12,251:,:]])

X_H2O_tensor = X_H2O_tensor_copy[0,:,:]
for i in range(1,12):
    X_H2O_tensor = torch.vstack([X_H2O_tensor, X_H2O_tensor_copy[i,:,:]])
X_H2O_tensor = torch.vstack([X_H2O_tensor, X_H2O_tensor_copy[12,251:,:]])

X_q_tensor = X_q_tensor_copy[0,:,:]
for i in range(1,12):
    X_q_tensor = torch.vstack([X_q_tensor, X_q_tensor_copy[i,:,:]])
X_q_tensor = torch.vstack([X_q_tensor, X_q_tensor_copy[12,251:,:]])

train_X_tensor = X_tensor
train_X_H2O_tensor = X_H2O_tensor
train_X_q_tensor = X_q_tensor

#%%
torch.save(train_X_tensor,'X_tensor_train')
torch.save(train_X_H2O_tensor,'X_H2O_tensor_train')
torch.save(train_X_q_tensor,'X_q_tensor_train')


#%%
device = torch.device('cpu')
train_X_tensor = torch.load('X_tensor_train')

#%%
X = np.load('test_params/X_tensor.npy')
X_q = np.load('test_params/X_q_tensor.npy')
X_H2O = np.load('test_params/X_H2O_tensor.npy')
q_2D = np.load('test_params/q_unet_tensor.npy')

X = X[:,:,:,0]
X_H2O = X_H2O[:,:,:,0]
X_q = X_q[:,:,:,0]
q_2D = q_2D[:,:,:,0]

X_copy = X.copy()
X_H2O_copy = X_H2O.copy()
X_q_copy = X_q.copy()

#%%
X[:,2,:] = X_copy[:,2,:] - q_2D[0,3,:] - K_min
X_H2O[:,2,:] = X_H2O_copy[:,2,:] - q_2D[0,3,:] - H_min
X_q[:,3,:] = X_q_copy[:,3,:] - q_2D[0,3,:] - K_min

X = np.transpose(X.reshape((37+37)*3,500))
X_H2O = np.transpose(X_H2O.reshape((37+37+50*3)*3,500))
X_q = np.transpose(X_q.reshape((37+37)*4,500))

#%%
X_tensor = torch.Tensor(X)     
X_H2O_tensor = torch.Tensor(X_H2O)  
X_q_tensor = torch.Tensor(X_q)                  

X_tensor_copy = torch.clone(X_tensor)
X_H2O_tensor_copy = torch.clone(X_H2O_tensor)
X_q_tensor_copy = torch.clone(X_q_tensor)

test_X_tensor = X_tensor
test_X_H2O_tensor = X_H2O_tensor
test_X_q_tensor = X_q_tensor

#%%
torch.save(test_X_tensor,'X_tensor_test')
torch.save(test_X_H2O_tensor,'X_H2O_tensor_test')
torch.save(test_X_q_tensor,'X_q_tensor_test')
#%%
device = torch.device('cpu')
train_X_tensor = torch.load('X_tensor_train')
test_X_tensor = torch.load('X_tensor_test')

#%%
q_train = np.load('train_params/q_tensor.npy')
q_test = np.load('test_params/q_tensor.npy')
q_train_copy = q_train.copy()
q_test_copy = q_test.copy()
q_train = np.transpose(q_train_copy)
q_test = np.transpose(q_test_copy)

#%%
#q_train = q_train//0.01 + abs(np.min(q_train_copy[:,:,:12])//0.01)
#q_test = q_test//0.01 + abs(np.min(q_train_copy[:,:,:12])//0.01)

tensor_y_train = torch.Tensor(q_train)
tensor_y_test = torch.Tensor(q_test)
train_y = tensor_y_train[0,:,:]
for i in range(1,12):
    train_y = torch.vstack([train_y, tensor_y_train[i,:,:]])
train_y = torch.vstack([train_y, tensor_y_train[12,251:,:]])   
test_y = tensor_y_test[0,:,:]

#%%
train_data = TensorDataset(train_X_tensor,train_y)
train_data_H2O = TensorDataset(train_X_H2O_tensor,train_y)
train_data_q = TensorDataset(train_X_q_tensor,train_y)

test_data = TensorDataset(test_X_tensor,test_y)
test_data_q = TensorDataset(test_X_q_tensor,test_y)
test_data_H2O = TensorDataset(test_X_H2O_tensor,test_y)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100)

train_loader_q = DataLoader(train_data_q, batch_size=8, shuffle=True)
test_loader_q = DataLoader(test_data_q, batch_size=100)

train_loader_H2O = DataLoader(train_data_H2O, batch_size=8, shuffle=True)
test_loader_H2O = DataLoader(test_data_H2O, batch_size=100)

#%%
torch.save(train_loader,'train_loader_X_3D_bin')
torch.save(test_loader,'test_loader_X_3D_bin')
torch.save(train_loader_H2O,'train_loader_XH2O_3D')
torch.save(test_loader_H2O,'test_loader_XH2O_3D')
torch.save(train_loader_q,'train_loader_Xq_3D')
torch.save(test_loader_q,'test_loader_Xq_3D')
