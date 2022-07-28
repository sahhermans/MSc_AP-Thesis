# Setup
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
#from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
from torchsummary import summary
#from PIL import Image
#import pandas as pd
#import h5py
from datetime import datetime
#from sklearn.model_selection import KFold
from torch.nn.parameter import Parameter 
import p3DNets
#import CoordConv
#import os, psutil;

#%%
"""
Cl = np.load('Cl_tensor.npy')
K = np.load('K_tensor.npy')
X = np.load('X_tensor.npy')
q = np.load('q_tensor.npy')

Cl_copy = Cl.copy()
K_copy = K.copy()

X_copy = X.copy()
q_copy = q.copy()

X = np.transpose(X_copy.reshape((37+37)*3,1000))
q = np.transpose(q_copy)

tensor_x = torch.Tensor(X) # transform to torch tensor                          #.requires_grad = True
tensor_y = torch.Tensor(q)

#tensor_Cl = torch.Tensor(Cl)
#tensor_K = torch.Tensor(K)
    
Cl[:,0,:] = np.round((Cl_copy[:,0,:] + 14.4596)//(3.6149/19.9))
Cl[:,1,:] = np.round((Cl_copy[:,1,:] + 14.4596)//(3.6149/19.9))
Cl[:,2,:] = np.round((Cl_copy[:,2,:] + 35)//(3.6149/19.9))
  
K[:,0,:] = np.round((K_copy[:,0,:] + 14.4596)//(3.6149/19.9))
K[:,1,:] = np.round((K_copy[:,1,:] + 14.4596)//(3.6149/19.9))
K[:,2,:] = np.round((K_copy[:,2,:] + 35)//(3.6149/19.9))

#tensor_Cl = torch.Tensor(Cl)
#tensor_K = torch.Tensor(K)

x = np.vstack([Cl[:,0,:],K[:,0,:]]).astype(int)
y = np.vstack([Cl[:,1,:],K[:,1,:]]).astype(int)
z = np.vstack([Cl[:,2,:],K[:,2,:]]).astype(int)

indices = np.array([z,
                    x,
                    y])

x_Cl = Cl[:,0,:].astype(int)
y_Cl = Cl[:,1,:].astype(int)
z_Cl = Cl[:,2,:].astype(int)

x_K = K[:,0,:].astype(int)
y_K = K[:,1,:].astype(int)
z_K = K[:,2,:].astype(int)

values = np.hstack([-np.ones(37),np.ones(37)]).astype(int)

# if x_K == x_Cl, dan wat? coalesce!

zeros_indices = np.zeros(shape=np.shape(indices[:,:,0]))
zeros_values = np.zeros(shape=np.shape(values))

i = 0
idx_Cl = np.where(z_Cl[:,i] < 160)
idx_K = np.where(z_K[:,i] < 160)
    
indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                      np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i]]),
                    np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i]]),
                   np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i]])])
    
values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)

s_train = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, 160, 160, 160), dtype=torch.float64), dim=0)
#s = torch.stack([s for _ in range(1000)], dim=3)
  
for i in range(1,900):
    idx_Cl = np.where(z_Cl[:,i] < 160)
    idx_K = np.where(z_K[:,i] < 160)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                          np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i]]),
                        np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i]]),
                       np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
        
    s_train = torch.cat([s_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, 160, 160, 160), dtype=torch.float64), dim=0)], dim=0)

#%%
i = 900
idx_Cl = np.where(z_Cl[:,i] < 160)
idx_K = np.where(z_K[:,i] < 160)
    
indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])), 
                      np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i]]),
                    np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i]]),
                   np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i]])])
    
values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
s_test = torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, 160, 160, 160), dtype=torch.float64), dim=0)

for i in range(901,1000):
    idx_Cl = np.where(z_Cl[:,i] < 160)
    idx_K = np.where(z_K[:,i] < 160)
        
    indices_i = np.array([np.zeros(shape=(1,np.shape(idx_K)[1]+np.shape(idx_Cl)[1])),
                          np.hstack([z_Cl[idx_Cl,i],z_K[idx_K,i]]),
                        np.hstack([x_Cl[idx_Cl,i],x_K[idx_K,i]]),
                       np.hstack([y_Cl[idx_Cl,i],y_K[idx_K,i]])])
        
    values_i = np.transpose(np.hstack([-np.ones(np.shape(idx_Cl)[1]),np.ones(np.shape(idx_K)[1])])).astype(int)
        
    s_test = torch.cat([s_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_i[:,0,:], values_i, (1, 160, 160, 160), dtype=torch.float64), dim=0)], dim=0)

s_train.requires_grad = True
"""
#%%
def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
# Try using gpu instead of cpu
device = try_gpu()
"""
indices_q = np.expand_dims(np.linspace(0,127,128), axis = 0)

'
q_train = torch.unsqueeze(torch.sparse_coo_tensor(indices_q, q[0,:], (128,), dtype=torch.float64), dim=0)
for i in range(1,900):
    q_train = torch.cat([q_train,torch.unsqueeze(torch.sparse_coo_tensor(indices_q, q[i,:], (128,), dtype=torch.float64), dim=0)], dim=0)

q_test = torch.unsqueeze(torch.sparse_coo_tensor(indices_q, q[900,:], (128,), dtype=torch.float64), dim=0)
for i in range(901,1000):
    q_test = torch.cat([q_test,torch.unsqueeze(torch.sparse_coo_tensor(indices_q, q[i,:], (128,), dtype=torch.float64), dim=0)], dim=0)
'
tensor_y = torch.Tensor(q)

s_train = s_train.to(device=device, dtype=torch.float)
s_test = s_test.to(device=device, dtype=torch.float)

#s_train = torch.unsqueeze(s_train, dim = 1)
#s_test = torch.unsqueeze(s_test, dim = 1)

train_y = tensor_y[:900]
test_y = tensor_y[900:]

train_data = TensorDataset(s_train,train_y) # create your dataset
test_data = TensorDataset(s_test,test_y) # create your dataset

# Define data loaders used to iterate through dataset
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)
"""
#%%
"""
q_1D = np.load('q_tensor.npy')
q_1D_copy = q_1D.copy()
q_1D = np.transpose(q_1D_copy)

s_train = torch.load('s_train_d')
s_test = torch.load('s_test_d')

tensor_y = torch.Tensor(q_1D)

s_train_3D = s_train.to(device=device, dtype=torch.float)
s_test_3D = s_test.to(device=device, dtype=torch.float)
train_y = tensor_y[:900]
test_y = tensor_y[900:]
train_data = TensorDataset(s_train_3D,train_y) # create your dataset
test_data = TensorDataset(s_test_3D,test_y) # create your dataset
# Define data loaders used to iterate through dataset
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)
#%%
torch.save(train_loader,'train_loader_3D')
torch.save(test_loader,'test_loader_3D')
#%%
"""



#%%
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()
    
class Net(nn.Module):
    """
    16-layer CNN network with max pooling
    
    Args:
        in_channels: number of features of the input image ("depth of image")
        hidden_channels: number of hidden features ("depth of convolved images")
        out_features: number of features in output layer
    """
    
    def __init__(self, in_channels, hidden_channels, out_features):
        super(Net, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, hidden_channels[0], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(hidden_channels[0], hidden_channels[0], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu2 = nn.ReLU()
        self.max_pool1 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(hidden_channels[0], hidden_channels[1], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(hidden_channels[1], hidden_channels[1], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu4 = nn.ReLU()
        self.max_pool2 = nn.MaxPool3d(2)
        
        self.conv5 = nn.Conv3d(hidden_channels[1], hidden_channels[2], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv3d(hidden_channels[2], hidden_channels[2], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv3d(hidden_channels[2], hidden_channels[2], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu7 = nn.ReLU()
        self.max_pool3 = nn.MaxPool3d(2)

        self.conv8 = nn.Conv3d(hidden_channels[2], hidden_channels[3], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv3d(hidden_channels[3], hidden_channels[3], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv3d(hidden_channels[3], hidden_channels[3], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu10 = nn.ReLU()
        self.max_pool4 = nn.MaxPool3d(2)

        self.conv11 = nn.Conv3d(hidden_channels[3], hidden_channels[4], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv3d(hidden_channels[4], hidden_channels[4], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv3d(hidden_channels[4], hidden_channels[4], 
                                     kernel_size=3, stride=1, padding=1, 
                                     groups=1)
        self.relu13 = nn.ReLU()
        self.max_pool5 = nn.MaxPool3d(2)
        
        self.conv14 = nn.Conv3d(hidden_channels[4], hidden_channels[5], 
                                     kernel_size=(5,1,1), stride=1, padding=0)
        self.relu14 = nn.ReLU()
        
        self.linear1 = nn.Linear(5*5*hidden_channels[5], hidden_channels[6])

        self.fc = nn.Linear(hidden_channels[6], out_features)

    def forward(self, x):
        # Convolutional layer
        x = self.conv1(x)
        # Activation function
        x = self.relu1(x)
        x = self.conv2(x)
        # Activation function
        x = self.relu2(x)
        # Max pool
        x = self.max_pool1(x)
        
        x = self.conv3(x)
        # Activation function
        x = self.relu3(x)
        x = self.conv4(x)
        # Activation function
        x = self.relu4(x)
        # Max pool
        x = self.max_pool2(x)

        x = self.conv5(x)
        # Activation function
        x = self.relu5(x)
        x = self.conv6(x)
        # Activation function
        x = self.relu6(x)
        x = self.conv7(x)
        # Activation function
        x = self.relu7(x)
        # Max pool
        x = self.max_pool3(x)
        
        x = self.conv8(x)
        # Activation function
        x = self.relu8(x)
        x = self.conv9(x)
        # Activation function
        x = self.relu9(x)
        x = self.conv10(x)
        # Activation function
        x = self.relu10(x)
        # Max pool
        x = self.max_pool4(x)
        
        x = self.conv11(x)
        # Activation function
        x = self.relu11(x)
        x = self.conv12(x)
        # Activation function
        x = self.relu12(x)
        x = self.conv13(x)
        # Activation function
        x = self.relu13(x)
        # Max pool
        x = self.max_pool5(x)
        
        x = self.conv14(x)
        # Activation function
        x = self.relu14(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.linear1(x)
        # Fully connected layer
        x = self.fc(x)
        
        return x   
    
def custom_loss(y_pred, y_batch):
    
    loss = torch.mean(torch.abs(y_batch - y_pred))*128*4
    
    constraint = torch.sum(torch.abs(-4 - torch.sum(y_pred, axis = 1)))/(y_batch.shape[0])
    
    return (loss + constraint)

def custom_loss_RMSE(y_pred, y_batch):
    
    loss = torch.sqrt(torch.mean(torch.square(y_batch - y_pred)))*128*4
    
    constraint = torch.sqrt(torch.sum(torch.square(-4 - torch.sum(y_pred, axis = 1)))/(y_batch.shape[0]))
    
    return (loss + constraint)

def evaluate_accuracy(data_loader, net, device):
    """Evaluate accuracy of a model on the given data set."""
    net.eval()  #make sure network is in evaluation mode

    #init
    loss_sum = torch.tensor([0], dtype=torch.float32, device=device)
    con_sum = torch.tensor([0], dtype=torch.float32, device=device)
    n = 0

    for X, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device).to_dense(), y.to(device)
        with torch.no_grad():
         
            loss_sum += torch.sum(torch.abs(net(X) - y))/128
            con_sum += torch.sum(torch.abs(torch.sum(net(X), axis = 1) - torch.sum(y,axis = 1)))/128
            
            n += y.shape[0] #increases with the number of samples in the batch
    return loss_sum.item()/n, con_sum.item()/n


def train_test(epochs, net, device, train_loader, test_loader, optimizer, criterion):
    """Training and testing function of FCN"""
    
    startTime = datetime.now()
    
    # Define list to store losses and performances of each iteration
    train_losses = []
    train_losss = []
    test_losss = []
    train_cons = []
    test_cons = []
    
    for epoch in range(epochs):

        # Network in training mode and to device
        net.train()
        net.to(device)
        
        #for name, module in net.named_modules():
        #    if isinstance(module, torch.nn.Conv2d):
        #        module.weight = torch.nn.Parameter(module.weight.data.to_sparse())
        #        module.bias = torch.nn.Parameter(module.bias.data.to_sparse())

        # Training loop
        for i, (x_batch, y_batch) in enumerate(train_loader):

            # Set to same device
            
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.to_dense()
            
            # Set the gradients to zero
            optimizer.zero_grad()

            # Perform forward pass
            y_pred = net(x_batch)

            # Compute the loss
            loss = criterion(y_pred, y_batch)
            train_losses.append(loss.detach().cpu().numpy())
            
            # Backward computation and update
            loss.backward()
            optimizer.step()
            
            #print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
           
        print(datetime.now() - startTime)
        
        if epoch == 10: 
            learning_rate = 0.0001
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.5)
        elif epoch == 20:
            learning_rate = 0.00001
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.5)
        elif epoch == 30:
            learning_rate = 0.000001
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.5) 
                    
        if  epoch == 0 or epoch%5 == 0:
            # Compute train and test error
            train_loss, train_con = evaluate_accuracy(train_loader, net.to(device), device)
            test_loss, test_con = evaluate_accuracy(test_loader, net.to(device), device)
            
            # Development of performance
            train_losss.append(train_loss)
            test_losss.append(test_loss)
            
            train_cons.append(train_con)
            test_cons.append(test_con)
    
            # Print performance
            print('Epoch: {:.0f}'.format(epoch+1))
            print('Loss on train set: {:.7f}'.format(train_loss))
            print('Loss on test set: {:.7f}'.format(test_loss))
            print('Con on train set: {:.7f}'.format(train_con))
            print('Con on test set: {:.7f}'.format(test_con))
            print('')

    print(datetime.now() - startTime)
    
    return net, train_losses, train_losss, test_losss, train_cons, test_cons

#%%
net = torch.load('net_noAnoP_lrelu_nieuw')

test_loader = torch.load('../loaders/d_96/test_loader_d_3D')
train_loader = torch.load('../loaders/d_96/train_loader_d_3D_8')

data_loader = test_loader

net.eval()  #make sure network is in evaluation mode
net.to(device)

#init
loss_sum = torch.tensor([0], dtype=torch.float32, device=device)
con_sum = torch.tensor([0], dtype=torch.float32, device=device)
n = 0
results_list = np.empty(shape=(0))
y_list = np.empty(shape=(0))
peratomloss = np.zeros(shape=(500,128))

for i, (X, y) in enumerate(data_loader):
    # Copy the data to device.
    X, y = X.to(device).to_dense(), y.to(device)
    with torch.no_grad():
     
        loss_sum += torch.sum(torch.abs(net(X) - y))/128
        con_sum += torch.sum(torch.abs(torch.sum(net(X), axis = 1) - torch.sum(y,axis = 1)))/128
                
        peratomloss[4*i:4*i+4,:] = torch.abs(net(X) - y).detach().cpu().numpy()
        
        n += y.shape[0] #increases with the number of samples in the batch
    
        results_list = np.append(net(X).detach().cpu().flatten(), results_list)
        y_list = np.append(y.detach().cpu().flatten(), y_list)
  
loss_sum = loss_sum.item()/n
con_sum = con_sum.item()/n

#%%
import pandas as pd 

# color coder 128
exclude = list(range(0, 6000))
del exclude[9:6879]
infile = pd.read_csv("dumpq.04.lammpstrj", skiprows = exclude, sep = ' ', header = None)
infile.columns = ['A','Type','x','y','z','q']

#%%
infile = infile[(infile['Type'] == 5)]
infile = infile[infile['z'] == max(infile['z'])]
infile = infile.sort_values('A')
n_list = np.arange(1,129)
infile = infile.assign(A = list(n_list))

#%%
array0 = pd.DataFrame(np.mean(peratomloss, axis=0))
array0.columns = ['q']
infile0 = infile.assign(q = list(array0['q']))#infile0 = pd.concat([infile,array0],axis = 1)
outfilegeoname = "dumpq.error.lammpstrj"

infileAsString = infile0.to_string(header=False,index=False,float_format = '%.6f')

with open('dumpq.04.lammpstrj', 'r') as fin:
    runscript = fin.read().splitlines(True)
beginning = runscript[:9]

with open(outfilegeoname, 'w') as fout:
    fout.writelines(beginning)
    fout.writelines(infileAsString)

#%%
import matplotlib.pyplot as plt
from collections import OrderedDict

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

color = ['blue','red','g','c','y','k']
linestyle = ['solid',linestyles['densely dashdotted']]

#%%
#results_list = np.load('results_list_noadapnopool0.npy')
y_list = np.load('y_list_noadapnopool0.npy')
#%%
#results = plt.hist(results_list,100,color = color[0],histtype='step',range=(-0.29,0.12))
y = plt.hist(y_list,100,color = color[1],histtype='step',range=(-0.29,0.12))
plt.ylim(0,6500)
plt.grid()
plt.xlim(-0.3,0.15)

#plt.legend(['NN','SCC-DFTB'], fontsize = 12, loc = 'upper center')
plt.xlabel(r'Partial charge [e]',fontsize=14)
plt.ylabel('Count [-]',fontsize=14)

#%%
count_results = sum(i >= 0 for i in results_list)
count_y = sum(i >= 0 for i in y_list)

countmin_results = sum(i <= 0 for i in results_list)
countmin_y = sum(i <= 0 for i in y_list)
#%%
plt.figure(2)
plt.plot(results[1][1:],results[0]/64000,color[0],linestyle = linestyle[0])
plt.plot(y[1][:-1],y[0]/64000,color[1],linestyle = linestyle[1])
plt.ylim(0,0.1)
plt.grid()
plt.xlim(-0.3,0.15)

plt.legend(['NN','SCC-DFTB'], fontsize = 12, loc = 'upper center')
plt.xlabel(r'Partial charge [e]',fontsize=14)
plt.ylabel('Count [-]',fontsize=14)


#%%
import scipy.integrate as integrate
resultsint = integrate.trapezoid(results[0]/64000,results[1][1:])
yint = integrate.trapezoid(y[0]/64000,y[1][1:])

#%%,density=True
#p.save('results_list_lrelu',results_list)
#np.save('y_list_lrelu',y_list)
#%%
in_channels = 1
hidden_channels = [4,8,16,32,64,128,256]
out_features = 336

#%%
#net = Net(in_channels, hidden_channels, out_features)
#summary(net, (1, 160, 160, 160), device='cpu') 
#%%
# Training parameters
learning_rate = 0.0001
epochs = 1
criterion = custom_loss
criterion_RMSE = custom_loss_RMSE

"""
#%% default VGG
hidden_channels_vgg = [4,8,16,32,64,128,256,128]
net_vgg = p3DNets.VGG16_3D(in_channels, hidden_channels_vgg,True,True)
optimizer_vgg = torch.optim.Adam(net_vgg.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_vgg, (1, 160, 160, 160), device='cpu') 
#%%
net_vgg.apply(reset_weights)
net_vgg_vgg, train_losses_vgg, train_losss_vgg, test_losss_vgg, train_cons_vgg, test_cons_vgg = train_test(epochs, net_vgg, device, train_loader, test_loader, optimizer_vgg, criterion)
np.savez('net_vgg_3D.npz', netd=net_vgg_vgg, trainlossd=train_losses_vgg, trainlosssd=train_losss_vgg, testlosssd=test_losss_vgg, traincond=train_cons_vgg, testcond=test_cons_vgg)

#%% no weight decay
hidden_channels_vgg_nodecay = [4,8,16,32,64,128,256,128]
net_vgg_nodecay = p3DNets.VGG16_3D(in_channels, hidden_channels_vgg_nodecay,True,True)
optimizer_vgg_nodecay = torch.optim.Adam(net_vgg_nodecay.parameters(), lr=learning_rate, weight_decay=0.0)
summary(net_vgg_nodecay, (1, 160, 160, 160), device='cpu') 
#%%
net_vgg_nodecay.apply(reset_weights)
net_vgg_nodecay_vgg_nodecay, train_losses_vgg_nodecay, train_losss_vgg_nodecay, test_losss_vgg_nodecay, train_cons_vgg_nodecay, test_cons_vgg_nodecay = train_test(epochs, net_vgg_nodecay, device, train_loader, test_loader, optimizer_vgg_nodecay, criterion)
np.savez('net_vgg_nodecay_3D.npz', netd=net_vgg_nodecay_vgg_nodecay, trainlossd=train_losses_vgg_nodecay, trainlosssd=train_losss_vgg_nodecay, testlosssd=test_losss_vgg_nodecay, traincond=train_cons_vgg_nodecay, testcond=test_cons_vgg_nodecay)

#%% average pool
hidden_channels_vgg_avepool = [4,8,16,32,64,128,256,128]
net_vgg_avepool = p3DNets.VGG16_3D(in_channels, hidden_channels_vgg_avepool,False,True)
optimizer_vgg_avepool = torch.optim.Adam(net_vgg_avepool.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_vgg_avepool, (1, 160, 160, 160), device='cpu') 

#%%
net_vgg_avepool.apply(reset_weights)
net_vgg_avepool_vgg_avepool, train_losses_vgg_avepool, train_losss_vgg_avepool, test_losss_vgg_avepool, train_cons_vgg_avepool, test_cons_vgg_avepool = train_test(epochs, net_vgg_avepool, device, train_loader, test_loader, optimizer_vgg_avepool, criterion)
np.savez('net_vgg_avepool_3D.npz', netd=net_vgg_avepool_vgg_avepool, trainlossd=train_losses_vgg_avepool, trainlosssd=train_losss_vgg_avepool, testlosssd=test_losss_vgg_avepool, traincond=train_cons_vgg_avepool, testcond=test_cons_vgg_avepool)

#%% activation function
hidden_channels_vgg_act = [4,8,16,32,64,128,256,128]
net_vgg_act = p3DNets.VGG16_3D(in_channels, hidden_channels_vgg_act,True,False)
optimizer_vgg_act = torch.optim.Adam(net_vgg_act.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_vgg_act, (1, 160, 160, 160), device='cpu') 
#%%
net_vgg_act.apply(reset_weights)
net_vgg_act_vgg_act, train_losses_vgg_act, train_losss_vgg_act, test_losss_vgg_act, train_cons_vgg_act, test_cons_vgg_act = train_test(epochs, net_vgg_act, device, train_loader, test_loader, optimizer_vgg_act, criterion)
np.savez('net_vgg_act_3D.npz', netd=net_vgg_act_vgg_act, trainlossd=train_losses_vgg_act, trainlosssd=train_losss_vgg_act, testlosssd=test_losss_vgg_act, traincond=train_cons_vgg_act, testcond=test_cons_vgg_act)

#%% avepool at end
hidden_channels_vgg_endave = [4,8,16,32,64,128,128,128]
net_vgg_endave = p3DNets.VGG16_3D_endave(in_channels, hidden_channels_vgg_endave,True,True)
optimizer_vgg_endave = torch.optim.Adam(net_vgg_endave.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_vgg_endave, (1, 160, 160, 160), device='cpu') 
#%%
net_vgg_endave.apply(reset_weights)
net_vgg_endave_vgg_endave, train_losses_vgg_endave, train_losss_vgg_endave, test_losss_vgg_endave, train_cons_vgg_endave, test_cons_vgg_endave = train_test(epochs, net_vgg_endave, device, train_loader, test_loader, optimizer_vgg_endave, criterion)
np.savez('net_vgg_endave_3D.npz', netd=net_vgg_endave_vgg_endave, trainlossd=train_losses_vgg_endave, trainlosssd=train_losss_vgg_endave, testlosssd=test_losss_vgg_endave, traincond=train_cons_vgg_endave, testcond=test_cons_vgg_endave)

#%% large VGG
hidden_channels_vgg = [4,16,32,64,128,128,256,128]
net_vgg = p3DNets.VGG16_3D(in_channels, hidden_channels_vgg,True,True)
optimizer_vgg = torch.optim.Adam(net_vgg.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_vgg, (1, 160, 160, 160), device='cpu') 
#%%
net_vgg.apply(reset_weights)
net_vgg_vgg, train_losses_vgg, train_losss_vgg, test_losss_vgg, train_cons_vgg, test_cons_vgg = train_test(epochs, net_vgg, device, train_loader, test_loader, optimizer_vgg, criterion)
np.savez('net_vgg_3D.npz', netd=net_vgg_vgg, trainlossd=train_losses_vgg, trainlosssd=train_losss_vgg, testlosssd=test_losss_vgg, traincond=train_cons_vgg, testcond=test_cons_vgg)
"""

test_loader = torch.load('../loaders/d_96/test_loader_d_3D')
train_loader = torch.load('../loaders/d_96/train_loader_d_3D_8')

#%% default resnet
hidden_channels_res = [42,84,168,336]
net_res = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res,out_features)
optimizer_res = torch.optim.Adam(net_res.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res, (1, 96, 160, 160), device='cpu') 
#%%
net_res.apply(reset_weights)
net_res_res, train_losses_res, train_losss_res, test_losss_res, train_cons_res, test_cons_res = train_test(epochs, net_res, device, train_loader, test_loader, optimizer_res, criterion)
net_res_res = net_res_res.cpu()
torch.save(net_res_res,'net_res_res_test')
np.savez('net_res_3D_test.npz', trainlossd=train_losses_res, trainlosssd=train_losss_res, testlosssd=test_losss_res, traincond=train_cons_res, testcond=test_cons_res)

del net_res_res
del net_res

#%% default resnet batch 8
hidden_channels_res8 = [16,32,64,128]
net_res8 = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res8)
optimizer_res8 = torch.optim.Adam(net_res8.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_res8, (1, 160, 160, 160), device='cpu') 
#%%
net_res8.apply(reset_weights)
net_res8_res8, train_losses_res8, train_losss_res8, test_losss_res8, train_cons_res8, test_cons_res8 = train_test(epochs, net_res8, device, train_loader_8, test_loader, optimizer_res8, criterion)
net_res8_res8 = net_res8_res8.cpu()
torch.save(net_res8_res8,'net_res8_res8')
np.savez('net_res8_3D.npz', trainlossd=train_losses_res8, trainlosssd=train_losss_res8, testlosssd=test_losss_res8, traincond=train_cons_res8, testcond=test_cons_res8)

test_loader_H2O = torch.load('../loaders/H2O/test_loader_H2O_3D')
train_loader_H2O = torch.load('../loaders/H2O/train_loader_H2O_3D')

#%% H2O 
net_res_H2O = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res)
optimizer_res_H2O = torch.optim.Adam(net_res_H2O.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_res_H2O, (1, 160, 160, 160), device='cpu') 
#%%
net_res_H2O.apply(reset_weights)
net_res_H2O_res_H2O, train_losses_res_H2O, train_losss_res_H2O, test_losss_res_H2O, train_cons_res_H2O, test_cons_res_H2O = train_test(epochs, net_res_H2O, device, train_loader_H2O, test_loader_H2O, optimizer_res_H2O, criterion)
net_res_H2O_res_H2O = net_res_H2O_res_H2O.cpu()
torch.save(net_res_H2O_res_H2O,'net_res_H2O_res_H2O')
np.savez('net_res_H2O_3D.npz', trainlossd=train_losses_res_H2O, trainlosssd=train_losss_res_H2O, testlosssd=test_losss_res_H2O, traincond=train_cons_res_H2O, testcond=test_cons_res_H2O)

del net_res_H2O_res_H2O
del net_res_H2O
del test_loader_H2O
del train_loader_H2O

test_loader_xyz = torch.load('../loaders/xyz/test_loader_xyz_3D')
train_loader_xyz = torch.load('../loaders/xyz/train_loader_xyz_3D')

#%% xyz
net_res_xyz = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res)
optimizer_res_xyz = torch.optim.Adam(net_res_xyz.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_res_xyz, (1, 160, 160, 160), device='cpu') 
#%%
net_res_xyz.apply(reset_weights)
net_res_xyz_res_xyz, train_losses_res_xyz, train_losss_res_xyz, test_losss_res_xyz, train_cons_res_xyz, test_cons_res_xyz = train_test(epochs, net_res_xyz, device, train_loader_xyz, test_loader_xyz, optimizer_res_xyz, criterion)
net_res_xyz_res_xyz = net_res_xyz_res_xyz.cpu()
torch.save(net_res_xyz_res_xyz,'net_res_xyz_res_xyz')
np.savez('net_res_xyz_3D.npz', trainlossd=train_losses_res_xyz, trainlosssd=train_losss_res_xyz, testlosssd=test_losss_res_xyz, traincond=train_cons_res_xyz, testcond=test_cons_res_xyz)

del test_loader_xyz
del train_loader_xyz
del net_res_xyz_res_xyz
del net_res_xyz
test_loader_o = torch.load('../loaders/o/test_loader_o_3D')
train_loader_o = torch.load('../loaders/o/train_loader_o_3D')

#%% ongescaled
net_res_ongescaled = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res)
optimizer_res_ongescaled = torch.optim.Adam(net_res_ongescaled.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_res_ongescaled, (1, 160, 160, 160), device='cpu') 
#%%
net_res_ongescaled.apply(reset_weights)
net_res_ongescaled_res_ongescaled, train_losses_res_ongescaled, train_losss_res_ongescaled, test_losss_res_ongescaled, train_cons_res_ongescaled, test_cons_res_ongescaled = train_test(epochs, net_res_ongescaled, device, train_loader_o, test_loader_o, optimizer_res_ongescaled, criterion)
net_res_ongescaled_res_ongescaled = net_res_ongescaled_res_ongescaled.cpu()
torch.save(net_res_ongescaled_res_ongescaled,'net_res_ongescaled_res_ongescaled')
np.savez('net_res_ongescaled_3D.npz', trainlossd=train_losses_res_ongescaled, trainlosssd=train_losss_res_ongescaled, testlosssd=test_losss_res_ongescaled, traincond=train_cons_res_ongescaled, testcond=test_cons_res_ongescaled)

del test_loader_o
del train_loader_o
del net_res_o_res_o
del net_res_o

"""

test_loader_30d = torch.load('../loaders/d30/test_loader_d30_3D')
train_loader_30d = torch.load('../loaders/d30/train_loader_d30_3D')


#%% default resnet 30d
hidden_channels_res30d = [16,32,64,128]
net_res30d = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res30d)
optimizer_res30d = torch.optim.Adam(net_res30d.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_res30d, (1, 160, 160, 160), device='cpu') 
#%%
net_res30d.apply(reset_weights)
net_res30d_res30d, train_losses_res30d, train_losss_res30d, test_losss_res30d, train_cons_res30d, test_cons_res30d = train_test(epochs, net_res30d, device, train_loader_30d, test_loader_30d, optimizer_res30d, criterion)
net_res30d_res30d = net_res30d_res30d.cpu()
torch.save(net_30d_30d,'net_30d_30d')
np.savez('net_res30d_3D.npz', trainlossd=train_losses_res30d, trainlosssd=train_losss_res30d, testlosssd=test_losss_res30d, traincond=train_cons_res30d, testcond=test_cons_res30d)

del test_loader_30d
del train_loader_30d
del net_res30d_res30d
del net_res30d
test_loader_64 = torch.load('../loaders/d_64/test_loader_64_3D')
train_loader_64 = torch.load('../loaders/d_64/train_loader_64_3D')

#%% default resnet z64
hidden_channels_res64 = [16,32,64,128]
net_res64 = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res64)
optimizer_res64 = torch.optim.Adam(net_res64.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_res64, (1, 160, 160, 160), device='cpu') 
#%%
net_res64.apply(reset_weights)
net_res64_res64, train_losses_res64, train_losss_res64, test_losss_res64, train_cons_res64, test_cons_res64 = train_test(epochs, net_res64, device, train_loader_64, test_loader_64, optimizer_res64, criterion)
net_res64_res64 = net_res64_res64.cpu()
torch.save(net_res64_res64,'net_res64_res64')
np.savez('net_res64_3D.npz', trainlossd=train_losses_res64, trainlosssd=train_losss_res64, testlosssd=test_losss_res64, traincond=train_cons_res64, testcond=test_cons_res64)

del net_res64_res64
del net_res64
del test_loader_64
del train_loader_64

test_loader_H2Of = torch.load('../loaders/H2O/test_loader_H2O_f_3D')
train_loader_H2Of = torch.load('../loaders/H2O/train_loader_H2O_f_3D')

#%% H2O full 
net_res_H2O_full = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res)
optimizer_res_H2O_full = torch.optim.Adam(net_res_H2O_full.parameters(), lr=learning_rate, weight_decay=0.5)
#summary(net_res_H2O_full, (1, 160, 160, 160), device='cpu') 
#%%
net_res_H2O_full.apply(reset_weights)
net_res_H2O_full_res_H2O_full, train_losses_res_H2O_full, train_losss_res_H2O_full, test_losss_res_H2O_full, train_cons_res_H2O_full, test_cons_res_H2O_full = train_test(epochs, net_res_H2O_full, device, train_loader_H2Of, test_loader_H2Of, optimizer_res_H2O_full, criterion)
net_res_H2O_full_res_H2O_full = net_res_H2O_full_res_H2O_full.cpu()
torch.save(net_res_H2O_full_res_H2O_full,'net_res_H2O_full_res_H2O_full')
np.savez('net_res_H2O_full_3D.npz', trainlossd=train_losses_res_H2O_full, trainlosssd=train_losss_res_H2O_full, testlosssd=test_losss_res_H2O_full, traincond=train_cons_res_H2O_full, testcond=test_cons_res_H2O_full)


"""
#%% small 4 resnet
hidden_channels_4 = [16,32,64,128]
net_4 = p3DNets.ResNet4_3D_d(in_channels, hidden_channels_4)
optimizer_4 = torch.optim.Adam(net_4.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_4, (1, 96, 160, 160), device='cpu') 
#%%
net_4.apply(reset_weights)
net_4_4, train_losses_4, train_losss_4, test_losss_4, train_cons_4, test_cons_4 = train_test(epochs, net_4, device, train_loader, test_loader, optimizer_4, criterion)
net_4_4 = net_4_4.cpu()
torch.save(net_4_4,'net_4_4_test')
np.savez('net_4_3D_test.npz', trainlossd=train_losses_4, trainlosssd=train_losss_4, testlosssd=test_losss_4, traincond=train_cons_4, testcond=test_cons_4)

del net_4_4
del net_4

#%% no adap nopool deep
hidden_channels_noadapnopool_deep = [16,32,64,128,128]
net_noadapnopool_deep = p3DNets.ResNet18_3D_noadaptive_nopool(in_channels, hidden_channels_noadapnopool_deep)
optimizer_noadapnopool_deep = torch.optim.Adam(net_noadapnopool_deep.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadapnopool_deep, (1, 96, 160, 160), device='cpu') 
#%%
net_noadapnopool_deep.apply(reset_weights)
net_noadapnopool_deep_noadapnopool_deep, train_losses_noadapnopool_deep, train_losss_noadapnopool_deep, test_losss_noadapnopool_deep, train_cons_noadapnopool_deep, test_cons_noadapnopool_deep = train_test(epochs, net_noadapnopool_deep, device, train_loader, test_loader, optimizer_noadapnopool_deep, criterion)
net_noadapnopool_deep_noadapnopool_deep = net_noadapnopool_deep_noadapnopool_deep.cpu()
torch.save(net_noadapnopool_deep_noadapnopool_deep,'net_noadapnopool_deep')
np.savez('net_noadapnopool_deep_3D.npz', trainlossd=train_losses_noadapnopool_deep, trainlosssd=train_losss_noadapnopool_deep, testlosssd=test_losss_noadapnopool_deep, traincond=train_cons_noadapnopool_deep, testcond=test_cons_noadapnopool_deep)

del net_noadapnopool_deep
del net_noadapnopool_deep_noadapnopool_deep

#%% no adap extra linear zonder relu
hidden_channels_noadaplin128 = [16,32,64,128,128]
net_noadaplin128 = p3DNets.ResNet18_3D_noadaptive_extralin(in_channels, hidden_channels_noadaplin128)
optimizer_noadaplin128 = torch.optim.Adam(net_noadaplin128.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadaplin128, (1, 96, 160, 160), device='cpu') 
#%%
net_noadaplin128.apply(reset_weights)
net_noadaplin128_noadaplin128, train_losses_noadaplin128, train_losss_noadaplin128, test_losss_noadaplin128, train_cons_noadaplin128, test_cons_noadaplin128 = train_test(epochs, net_noadaplin128, device, train_loader, test_loader, optimizer_noadaplin128, criterion)
net_noadaplin128_noadaplin128 = net_noadaplin128_noadaplin128.cpu()
torch.save(net_noadaplin128_noadaplin128,'net_noadaplin128')
np.savez('net_noadaplin128_3D.npz', trainlossd=train_losses_noadaplin128, trainlosssd=train_losss_noadaplin128, testlosssd=test_losss_noadaplin128, traincond=train_cons_noadaplin128, testcond=test_cons_noadaplin128)

del net_noadaplin128_noadaplin128
del net_noadaplin128

#%% no adaptive no pooling RMSE *128
hidden_channels_res_nopool = [16,32,64,128,128]
net_res_nopool = p3DNets.ResNet18_3D_noadaptive_nopool(in_channels, hidden_channels_res_nopool)
optimizer_res_nopool = torch.optim.Adam(net_res_nopool.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_nopool, (1, 96, 160, 160), device='cpu') 
#%%
net_res_nopool.apply(reset_weights)
net_res_nopool_res_nopool, train_losses_res_nopool, train_losss_res_nopool, test_losss_res_nopool, train_cons_res_nopool, test_cons_res_nopool = train_test(epochs, net_res_nopool, device, train_loader, test_loader, optimizer_res_nopool, criterion_RMSE)
net_nopool_nopool = net_res_nopool_res_nopool.cpu()
torch.save(net_nopool_nopool,'net_noadap_RMSE_128')
np.savez('net_res_noadap_RMSE_128.npz', netd=net_res_nopool_res_nopool, trainlossd=train_losses_res_nopool, trainlosssd=train_losss_res_nopool, testlosssd=test_losss_res_nopool, traincond=train_cons_res_nopool, testcond=test_cons_res_nopool)

del net_res_nopool_res_nopool
del net_nopool_nopool
del net_res_nopool

#%% no adaptive no pooling RMSE orig
hidden_channels_res_nopool = [16,32,64,128,128]
net_res_nopool = p3DNets.ResNet18_3D_noadaptive_nopool(in_channels, hidden_channels_res_nopool)
optimizer_res_nopool = torch.optim.Adam(net_res_nopool.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_nopool, (1, 96, 160, 160), device='cpu') 
#%%
net_res_nopool.apply(reset_weights)
net_res_nopool_res_nopool, train_losses_res_nopool, train_losss_res_nopool, test_losss_res_nopool, train_cons_res_nopool, test_cons_res_nopool = train_test(epochs, net_res_nopool, device, train_loader, test_loader, optimizer_res_nopool, criterion_RMSE)
net_nopool_nopool = net_res_nopool_res_nopool.cpu()
torch.save(net_nopool_nopool,'net_noadap_RMSE_orig')
np.savez('net_res_noadap_RMSE_orig.npz', netd=net_res_nopool_res_nopool, trainlossd=train_losses_res_nopool, trainlosssd=train_losss_res_nopool, testlosssd=test_losss_res_nopool, traincond=train_cons_res_nopool, testcond=test_cons_res_nopool)

del net_res_nopool_res_nopool
del net_nopool_nopool
del net_res_nopool

#%% no adap nopool 12+
del train_loader
del test_loader
#%%
hidden_channels_noadapnopool0 = [16,32,64,128]
net_noadapnopool0 = p3DNets.ResNet18_3D_noadaptive_nopool(in_channels, hidden_channels_noadapnopool0)
optimizer_noadapnopool0 = torch.optim.Adam(net_noadapnopool0.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadapnopool0, (1, 96, 160, 160), device='cpu') 
#%%
net_noadapnopool0.apply(reset_weights)
net_noadapnopool0_noadapnopool0, train_losses_noadapnopool0, train_losss_noadapnopool0, test_losss_noadapnopool0, train_cons_noadapnopool0, test_cons_noadapnopool0 = train_test(epochs, net_noadapnopool0, device, train_loader, test_loader, optimizer_noadapnopool0, criterion)
net_noadapnopool0_noadapnopool0 = net_noadapnopool0_noadapnopool0.cpu()
torch.save(net_noadapnopool0_noadapnopool0,'net_noadapnopool0_12+')
np.savez('net_noadapnopool0_3D_12+.npz', trainlossd=train_losses_noadapnopool0, trainlosssd=train_losss_noadapnopool0, testlosssd=test_losss_noadapnopool0, traincond=train_cons_noadapnopool0, testcond=test_cons_noadapnopool0)

#%% no adap nopool lrelu
hidden_channels_noadapnopool0 = [16,32,64,128]
net_noadapnopool0 = p3DNets.ResNet18_3D_noadaptive_nopool_lrelu(in_channels, hidden_channels_noadapnopool0)
optimizer_noadapnopool0 = torch.optim.Adam(net_noadapnopool0.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadapnopool0, (1, 96, 160, 160), device='cpu') 
#%%
net_noadapnopool0.apply(reset_weights)
net_noadapnopool0_noadapnopool0, train_losses_noadapnopool0, train_losss_noadapnopool0, test_losss_noadapnopool0, train_cons_noadapnopool0, test_cons_noadapnopool0 = train_test(epochs, net_noadapnopool0, device, train_loader, test_loader, optimizer_noadapnopool0, criterion)
net_noadapnopool0_noadapnopool0 = net_noadapnopool0_noadapnopool0.cpu()
torch.save(net_noadapnopool0_noadapnopool0,'net_noadapnopool0_lrelu')
np.savez('net_noadapnopool0_lrelu_3D.npz', trainlossd=train_losses_noadapnopool0, trainlosssd=train_losss_noadapnopool0, testlosssd=test_losss_noadapnopool0, traincond=train_cons_noadapnopool0, testcond=test_cons_noadapnopool0)

#%% CoordConv
hidden_channels = [16,32,64,128,128]
net_CoordConv = CoordConv.Net_CoordConv(in_channels, hidden_channels)
optimizer_CoordConv = torch.optim.Adam(net_CoordConv.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_CoordConv, (1, 96, 160, 160), device='cpu') 








#%% CoordConv torch
hidden_channels_CoordConv_torch = [16,32,64,128,128]
net_CoordConv_torch = CoordConv.Net_CoordConv(in_channels, hidden_channels_CoordConv_torch)
optimizer_CoordConv_torch = torch.optim.Adam(net_CoordConv_torch.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_CoordConv_torch, (1, 96, 160, 160), device='cpu') 
#%%
net_CoordConv_torch.apply(reset_weights)
net_CoordConv_torch_CoordConv_torch, train_losses_CoordConv_torch, train_losss_CoordConv_torch, test_losss_CoordConv_torch, train_cons_CoordConv_torch, test_cons_CoordConv_torch = train_test(epochs, net_CoordConv_torch, device, train_loader, test_loader, optimizer_CoordConv_torch, criterion)
net_CoordConv_torch_CoordConv_torch = net_CoordConv_torch_CoordConv_torch.cpu()
torch.save(net_CoordConv_torch_CoordConv_torch,'net_CoordConv_torch')
np.savez('net_CoordConv_torch.npz', trainlossd=train_losses_CoordConv_torch, trainlosssd=train_losss_CoordConv_torch, testlosssd=test_losss_CoordConv_torch, traincond=train_cons_CoordConv_torch, testcond=test_cons_CoordConv_torch)

del net_CoordConv_torch_CoordConv_torch
del net_CoordConv_torch

#%% CoordConv keras
hidden_channels_CoordConv_keras = [16,32,64,128,128]
net_CoordConv_keras = CoordConv.Net_CoordConv_keras(in_channels, hidden_channels_CoordConv_keras)
optimizer_CoordConv_keras = keras.optim.Adam(net_CoordConv_keras.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_CoordConv_keras, (1, 96, 160, 160), device='cpu') 
#%%
net_CoordConv_keras.apply(reset_weights)
net_CoordConv_keras_CoordConv_keras, train_losses_CoordConv_keras, train_losss_CoordConv_keras, test_losss_CoordConv_keras, train_cons_CoordConv_keras, test_cons_CoordConv_keras = train_test(epochs, net_CoordConv_keras, device, train_loader, test_loader, optimizer_CoordConv_keras, criterion)
net_CoordConv_keras_CoordConv_keras = net_CoordConv_keras_CoordConv_keras.cpu()
keras.save(net_CoordConv_keras_CoordConv_keras,'net_CoordConv_keras')
np.savez('net_CoordConv_keras.npz', trainlossd=train_losses_CoordConv_keras, trainlosssd=train_losss_CoordConv_keras, testlosssd=test_losss_CoordConv_keras, traincond=train_cons_CoordConv_keras, testcond=test_cons_CoordConv_keras)

del net_CoordConv_keras_CoordConv_keras
del net_CoordConv_keras

#%% CoordConv keras circular
hidden_channels_CoordConv_keras_circular = [16,32,64,128,128]
net_CoordConv_keras_circular = CoordConv.Net_CoordConv_keras_circular(in_channels, hidden_channels_CoordConv_keras_circular)
optimizer_CoordConv_keras_circular = keras.optim.Adam(net_CoordConv_keras_circular.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_CoordConv_keras_circular, (1, 96, 160, 160), device='cpu') 
#%%
net_CoordConv_keras_circular.apply(reset_weights)
net_CoordConv_keras_circular_CoordConv_keras_circular, train_losses_CoordConv_keras_circular, train_losss_CoordConv_keras_circular, test_losss_CoordConv_keras_circular, train_cons_CoordConv_keras_circular, test_cons_CoordConv_keras_circular = train_test(epochs, net_CoordConv_keras_circular, device, train_loader, test_loader, optimizer_CoordConv_keras_circular, criterion)
net_CoordConv_keras_circular_CoordConv_keras_circular = net_CoordConv_keras_circular_CoordConv_keras_circular.cpu()
keras.save(net_CoordConv_keras_circular_CoordConv_keras_circular,'net_CoordConv_keras_circular')
np.savez('net_CoordConv_keras_circular.npz', trainlossd=train_losses_CoordConv_keras_circular, trainlosssd=train_losss_CoordConv_keras_circular, testlosssd=test_losss_CoordConv_keras_circular, traincond=train_cons_CoordConv_keras_circular, testcond=test_cons_CoordConv_keras_circular)

del net_CoordConv_keras_circular_CoordConv_keras_circular
del net_CoordConv_keras_circular

#%% CoordConv noAnoP torch
hidden_channels_CoordConv_torchnoAnoP = [16,32,64,128]
net_CoordConv_torchnoAnoP = CoordConv.ResNet18_noAnoP_CoordConv_torch(in_channels, hidden_channels_CoordConv_torchnoAnoP)
optimizer_CoordConv_torchnoAnoP = torch.optim.Adam(net_CoordConv_torchnoAnoP.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_CoordConv_torchnoAnoP, (1, 96, 160, 160), device='cpu') 
#%%
net_CoordConv_torchnoAnoP.apply(reset_weights)
net_CoordConv_torchnoAnoP_CoordConv_torchnoAnoP, train_losses_CoordConv_torchnoAnoP, train_losss_CoordConv_torchnoAnoP, test_losss_CoordConv_torchnoAnoP, train_cons_CoordConv_torchnoAnoP, test_cons_CoordConv_torchnoAnoP = train_test(epochs, net_CoordConv_torchnoAnoP, device, train_loader, test_loader, optimizer_CoordConv_torchnoAnoP, criterion)
net_CoordConv_torchnoAnoP_CoordConv_torchnoAnoP = net_CoordConv_torchnoAnoP_CoordConv_torchnoAnoP.cpu()
torch.save(net_CoordConv_torchnoAnoP_CoordConv_torchnoAnoP,'net_CoordConv_torchnoAnoP')
np.savez('net_CoordConv_torchnoAnoP.npz', trainlossd=train_losses_CoordConv_torchnoAnoP, trainlosssd=train_losss_CoordConv_torchnoAnoP, testlosssd=test_losss_CoordConv_torchnoAnoP, traincond=train_cons_CoordConv_torchnoAnoP, testcond=test_cons_CoordConv_torchnoAnoP)

del net_CoordConv_torchnoAnoP_CoordConv_torchnoAnoP
del net_CoordConv_torchnoAnoP

#%% CoordConv keras noAnoP
hidden_channels_CoordConv_kerasnoAnoP = [16,32,64,128]
net_CoordConv_kerasnoAnoP = CoordConv.ResNet18_noAnoP_CoordConv_keras(in_channels, hidden_channels_CoordConv_kerasnoAnoP)
optimizer_CoordConv_kerasnoAnoP = keras.optim.Adam(net_CoordConv_kerasnoAnoP.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_CoordConv_kerasnoAnoP, (1, 96, 160, 160), device='cpu') 
#%%
net_CoordConv_kerasnoAnoP.apply(reset_weights)
net_CoordConv_kerasnoAnoP_CoordConv_kerasnoAnoP, train_losses_CoordConv_kerasnoAnoP, train_losss_CoordConv_kerasnoAnoP, test_losss_CoordConv_kerasnoAnoP, train_cons_CoordConv_kerasnoAnoP, test_cons_CoordConv_kerasnoAnoP = train_test(epochs, net_CoordConv_kerasnoAnoP, device, train_loader, test_loader, optimizer_CoordConv_kerasnoAnoP, criterion)
net_CoordConv_kerasnoAnoP_CoordConv_kerasnoAnoP = net_CoordConv_kerasnoAnoP_CoordConv_kerasnoAnoP.cpu()
keras.save(net_CoordConv_kerasnoAnoP_CoordConv_kerasnoAnoP,'net_CoordConv_kerasnoAnoP')
np.savez('net_CoordConv_kerasnoAnoP.npz', trainlossd=train_losses_CoordConv_kerasnoAnoP, trainlosssd=train_losss_CoordConv_kerasnoAnoP, testlosssd=test_losss_CoordConv_kerasnoAnoP, traincond=train_cons_CoordConv_kerasnoAnoP, testcond=test_cons_CoordConv_kerasnoAnoP)

del net_CoordConv_kerasnoAnoP_CoordConv_kerasnoAnoP
del net_CoordConv_kerasnoAnoP



#%% no adap nopool lrelu
hidden_channels_noAnoP_lrelu = [16,32,64,128]
net_noAnoP_lrelu = p3DNets.ResNet18_3D_noadaptive_nopool_lrelu(in_channels, hidden_channels_noAnoP_lrelu)
optimizer_noAnoP_lrelu = torch.optim.Adam(net_noAnoP_lrelu.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noAnoP_lrelu, (1, 96, 160, 160), device='cpu') 
#%%
net_noAnoP_lrelu.apply(reset_weights)
net_noAnoP_lrelu_noAnoP_lrelu, train_losses_noAnoP_lrelu, train_losss_noAnoP_lrelu, test_losss_noAnoP_lrelu, train_cons_noAnoP_lrelu, test_cons_noAnoP_lrelu = train_test(epochs, net_noAnoP_lrelu, device, train_loader, test_loader, optimizer_noAnoP_lrelu, criterion)
net_noAnoP_lrelu_noAnoP_lrelu = net_noAnoP_lrelu_noAnoP_lrelu.cpu()
torch.save(net_noAnoP_lrelu_noAnoP_lrelu,'net_noAnoP_lrelu')
np.savez('net_noAnoP_lrelu_3D.npz', trainlossd=train_losses_noAnoP_lrelu, trainlosssd=train_losss_noAnoP_lrelu, testlosssd=test_losss_noAnoP_lrelu, traincond=train_cons_noAnoP_lrelu, testcond=test_cons_noAnoP_lrelu)

del net_noAnoP_lrelu_noAnoP_lrelu
del net_noAnoP_lrelu



#%% no adap nopool small first kernel
hidden_channels_noAnoP_smallfirstkernel = [16,32,64,128]
net_noAnoP_smallfirstkernel = p3DNets.ResNet18_3D_noadaptive_nopool_smallfirstkernel(in_channels, hidden_channels_noAnoP_smallfirstkernel)
optimizer_noAnoP_smallfirstkernel = torch.optim.Adam(net_noAnoP_smallfirstkernel.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noAnoP_smallfirstkernel, (1, 96, 160, 160), device='cpu') 
#%%
net_noAnoP_smallfirstkernel.apply(reset_weights)
net_noAnoP_smallfirstkernel_noAnoP_smallfirstkernel, train_losses_noAnoP_smallfirstkernel, train_losss_noAnoP_smallfirstkernel, test_losss_noAnoP_smallfirstkernel, train_cons_noAnoP_smallfirstkernel, test_cons_noAnoP_smallfirstkernel = train_test(epochs, net_noAnoP_smallfirstkernel, device, train_loader, test_loader, optimizer_noAnoP_smallfirstkernel, criterion)
net_noAnoP_smallfirstkernel_noAnoP_smallfirstkernel = net_noAnoP_smallfirstkernel_noAnoP_smallfirstkernel.cpu()
torch.save(net_noAnoP_smallfirstkernel_noAnoP_smallfirstkernel,'net_noAnoP_smallfirstkernel')
np.savez('net_noAnoP_smallfirstkernel_3D.npz', trainlossd=train_losses_noAnoP_smallfirstkernel, trainlosssd=train_losss_noAnoP_smallfirstkernel, testlosssd=test_losss_noAnoP_smallfirstkernel, traincond=train_cons_noAnoP_smallfirstkernel, testcond=test_cons_noAnoP_smallfirstkernel)

del net_noAnoP_smallfirstkernel_noAnoP_smallfirstkernel
del net_noAnoP_smallfirstkernel








test_loader_ext = torch.load('test_loader_d_3D_extended')
train_loader_ext = torch.load('train_loader_d_3D_extended')

#%% no adap nopool ext
hidden_channels_noAnoP_ext = [21,42,84,168,336]
net_noAnoP_ext = p3DNets.ResNet18_3D_noadaptive_nopool(in_channels, hidden_channels_noAnoP_ext)
optimizer_noAnoP_ext = torch.optim.Adam(net_noAnoP_ext.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noAnoP_ext, (1, 96, 160, 160), device='cpu') 
#%%
net_noAnoP_ext.apply(reset_weights)
net_noAnoP_ext_noAnoP_ext, train_losses_noAnoP_ext, train_losss_noAnoP_ext, test_losss_noAnoP_ext, train_cons_noAnoP_ext, test_cons_noAnoP_ext = train_test(epochs, net_noAnoP_ext, device, train_loader_ext, test_loader_ext, optimizer_noAnoP_ext, criterion)
net_noAnoP_ext_noAnoP_ext = net_noAnoP_ext_noAnoP_ext.cpu()
torch.save(net_noAnoP_ext_noAnoP_ext,'net_noAnoP_ext')
np.savez('net_noAnoP_ext_3D.npz', trainlossd=train_losses_noAnoP_ext, trainlosssd=train_losss_noAnoP_ext, testlosssd=test_losss_noAnoP_ext, traincond=train_cons_noAnoP_ext, testcond=test_cons_noAnoP_ext)

del net_noAnoP_ext_noAnoP_ext
del net_noAnoP_ext
del test_loader_ext
del train_loader_ext

test_loader_ext_H2O = torch.load('test_loader_d_3D_extended_H2O')
train_loader_ext_H2O = torch.load('train_loader_d_3D_extended_H2O')

#%% no adap nopool ext H2O
hidden_channels_noAnoP_ext_H2O = [16,32,64,128]
net_noAnoP_ext_H2O = p3DNets.ResNet18_3D_noadaptive_nopool(in_channels, hidden_channels_noAnoP_ext_H2O)
optimizer_noAnoP_ext_H2O = torch.optim.Adam(net_noAnoP_ext_H2O.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noAnoP_ext_H2O, (1, 96, 160, 160), device='cpu') 
#%%
net_noAnoP_ext_H2O.apply(reset_weights)
net_noAnoP_ext_H2O_noAnoP_ext_H2O, train_losses_noAnoP_ext_H2O, train_losss_noAnoP_ext_H2O, test_losss_noAnoP_ext_H2O, train_cons_noAnoP_ext_H2O, test_cons_noAnoP_ext_H2O = train_test(epochs, net_noAnoP_ext_H2O, device, train_loader_ext_H2O, test_loader_ext_H2O, optimizer_noAnoP_ext_H2O, criterion)
net_noAnoP_ext_H2O_noAnoP_ext_H2O = net_noAnoP_ext_H2O_noAnoP_ext_H2O.cpu()
torch.save(net_noAnoP_ext_H2O_noAnoP_ext_H2O,'net_noAnoP_ext_H2O')
np.savez('net_noAnoP_ext_H2O_3D.npz', trainlossd=train_losses_noAnoP_ext_H2O, trainlosssd=train_losss_noAnoP_ext_H2O, testlosssd=test_losss_noAnoP_ext_H2O, traincond=train_cons_noAnoP_ext_H2O, testcond=test_cons_noAnoP_ext_H2O)

del net_noAnoP_ext_H2O_noAnoP_ext_H2O
del net_noAnoP_ext_H2O
del test_loader_ext_H2O
del train_loader_ext_H2O

"""

#%% no adaptive pooling
hidden_channels_res_nopool = [16,32,64,128,128]
net_res_nopool = p3DNets.ResNet18_3D_noadaptive_2(in_channels, hidden_channels_res_nopool)
optimizer_res_nopool = torch.optim.Adam(net_res_nopool.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_nopool, (1, 96, 160, 160), device='cpu') 
#%%
net_res_nopool.apply(reset_weights)
net_res_nopool_res_nopool, train_losses_res_nopool, train_losss_res_nopool, test_losss_res_nopool, train_cons_res_nopool, test_cons_res_nopool = train_test(epochs, net_res_nopool, device, train_loader, test_loader, optimizer_res_nopool, criterion)
net_nopool_nopool = net_res_nopool_res_nopool.cpu()
torch.save(net_nopool_nopool,'net_nopool_nopool_test')
np.savez('net_res_nopool_3D.npz', netd=net_res_nopool_res_nopool, trainlossd=train_losses_res_nopool, trainlosssd=train_losss_res_nopool, testlosssd=test_losss_res_nopool, traincond=train_cons_res_nopool, testcond=test_cons_res_nopool)

del net_res_nopool_res_nopool
del net_nopool_nopool
del net_res_nopool

#%% extra conv
hidden_channels_res_extraconv = [16,32,64,128,128]
net_res_extraconv = p3DNets.ResNet18_3D_extraconv(in_channels, hidden_channels_res_extraconv)
optimizer_res_extraconv = torch.optim.Adam(net_res_extraconv.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_extraconv, (1, 96, 160, 160), device='cpu') 
#%%
net_res_extraconv.apply(reset_weights)
net_res_extraconv_res_extraconv, train_losses_res_extraconv, train_losss_res_extraconv, test_losss_res_extraconv, train_cons_res_extraconv, test_cons_res_extraconv = train_test(epochs, net_res_extraconv, device, train_loader, test_loader, optimizer_res_extraconv, criterion)
net_extraconv_extraconv = net_res_extraconv_res_extraconv.cpu()
torch.save(net_extraconv_extraconv,'net_extraconv_extraconv_test')
np.savez('net_res_extraconv_3D.npz', netd=net_res_extraconv_res_extraconv, trainlossd=train_losses_res_extraconv, trainlosssd=train_losss_res_extraconv, testlosssd=test_losss_res_extraconv, traincond=train_cons_res_extraconv, testcond=test_cons_res_extraconv)

del net_res_extraconv_res_extraconv
del net_extraconv_extraconv
del net_res_extraconv

#%% extra linear
hidden_channels_res_extralinear = [16,32,64,128,128]
net_res_extralinear = p3DNets.ResNet18_3D_extralinear(in_channels, hidden_channels_res_extralinear)
optimizer_res_extralinear = torch.optim.Adam(net_res_extralinear.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_extralinear, (1, 96, 160, 160), device='cpu') 
#%%
net_res_extralinear.apply(reset_weights)
net_res_extralinear_res_extralinear, train_losses_res_extralinear, train_losss_res_extralinear, test_losss_res_extralinear, train_cons_res_extralinear, test_cons_res_extralinear = train_test(epochs, net_res_extralinear, device, train_loader, test_loader, optimizer_res_extralinear, criterion)
net_extralinear_extralinear = net_res_extralinear_res_extralinear.cpu()
torch.save(net_extralinear_extralinear,'net_extralinear_extralinear_test')
np.savez('net_res_extralinear_3D.npz', netd=net_res_extralinear_res_extralinear, trainlossd=train_losses_res_extralinear, trainlosssd=train_losss_res_extralinear, testlosssd=test_losss_res_extralinear, traincond=train_cons_res_extralinear, testcond=test_cons_res_extralinear)

del net_res_extralinear_res_extralinear
del net_extralinear_extralinear
del net_res_extralinear



#%% wide no adaptive resnet
hidden_channels_wide = [32,64,128,256,128]
net_wide = p3DNets.ResNet18_3D_noadaptive_2(in_channels, hidden_channels_wide)
optimizer_wide = torch.optim.Adam(net_wide.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_wide, (1, 96, 160, 160), device='cpu') 
#%%
net_wide.apply(reset_weights)
net_wide_wide, train_losses_wide, train_losss_wide, test_losss_wide, train_cons_wide, test_cons_wide = train_test(epochs, net_wide, device, train_loader, test_loader, optimizer_wide, criterion)
net_wide_wide = net_wide_wide.cpu()
torch.save(net_wide_wide,'net_wide_noadap')
np.savez('net_wide_3D_noadap.npz', trainlossd=train_losses_wide, trainlosssd=train_losss_wide, testlosssd=test_losss_wide, traincond=train_cons_wide, testcond=test_cons_wide)


#%% largekernel resnet
hidden_channels_largekernel = [16,32,64,128,128]
net_largekernel = p3DNets.ResNet18_3D_largekernel(in_channels, hidden_channels_largekernel)
optimizer_largekernel = torch.optim.Adam(net_largekernel.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_largekernel, (1, 96, 160, 160), device='cpu') 
#%%
net_largekernel.apply(reset_weights)
net_largekernel_largekernel, train_losses_largekernel, train_losss_largekernel, test_losss_largekernel, train_cons_largekernel, test_cons_largekernel = train_test(epochs, net_largekernel, device, train_loader, test_loader, optimizer_largekernel, criterion)
net_largekernel_largekernel = net_largekernel_largekernel.cpu()
torch.save(net_largekernel_largekernel,'net_largekernel_largekernel')
np.savez('net_largekernel_3D_test.npz', trainlossd=train_losses_largekernel, trainlosssd=train_losss_largekernel, testlosssd=test_losss_largekernel, traincond=train_cons_largekernel, testcond=test_cons_largekernel)



#%% no adap large kernel
hidden_channels_noadaplargekernel = [16,32,64,128]
net_noadaplargekernel = p3DNets.ResNet18_3D_noadaptive_largekernel(in_channels, hidden_channels_noadaplargekernel)
optimizer_noadaplargekernel = torch.optim.Adam(net_noadaplargekernel.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadaplargekernel, (1, 96, 160, 160), device='cpu') 
#%%
net_noadaplargekernel.apply(reset_weights)
net_noadaplargekernel_noadaplargekernel, train_losses_noadaplargekernel, train_losss_noadaplargekernel, test_losss_noadaplargekernel, train_cons_noadaplargekernel, test_cons_noadaplargekernel = train_test(epochs, net_noadaplargekernel, device, train_loader, test_loader, optimizer_noadaplargekernel, criterion)
net_noadaplargekernel_noadaplargekernel = net_noadaplargekernel_noadaplargekernel.cpu()
torch.save(net_noadaplargekernel_noadaplargekernel,'net_noadaplargekernel')
np.savez('net_noadaplargekernel_3D.npz', trainlossd=train_losses_noadaplargekernel, trainlosssd=train_losss_noadaplargekernel, testlosssd=test_losss_noadaplargekernel, traincond=train_cons_noadaplargekernel, testcond=test_cons_noadaplargekernel)

del net_noadaplargekernel_noadaplargekernel
del net_noadaplargekernel

#%% no adap large kernel shallow
hidden_channels_noadapshallow = [16,32,64]
net_noadapshallow = p3DNets.ResNet18_3D_noadaptive_shallow(in_channels, hidden_channels_noadapshallow)
optimizer_noadapshallow = torch.optim.Adam(net_noadapshallow.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadapshallow, (1, 96, 160, 160), device='cpu') 
#%%
net_noadapshallow.apply(reset_weights)
net_noadapshallow_noadapshallow, train_losses_noadapshallow, train_losss_noadapshallow, test_losss_noadapshallow, train_cons_noadapshallow, test_cons_noadapshallow = train_test(epochs, net_noadapshallow, device, train_loader, test_loader, optimizer_noadapshallow, criterion)
net_noadapshallow_noadapshallow = net_noadapshallow_noadapshallow.cpu()
torch.save(net_noadapshallow_noadapshallow,'net_noadapshallow_noadapshallow_test')
np.savez('net_noadapshallow_3D_test.npz', trainlossd=train_losses_noadapshallow, trainlosssd=train_losss_noadapshallow, testlosssd=test_losss_noadapshallow, traincond=train_cons_noadapshallow, testcond=test_cons_noadapshallow)

del net_noadapshallow_noadapshallow
del net_noadapshallow

#%% no adaptive pooling RMSE
hidden_channels_res_nopool = [16,32,64,128,128]
net_res_nopool = p3DNets.ResNet18_3D_noadaptive_2(in_channels, hidden_channels_res_nopool)
optimizer_res_nopool = torch.optim.Adam(net_res_nopool.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_nopool, (1, 96, 160, 160), device='cpu') 
#%%
net_res_nopool.apply(reset_weights)
net_res_nopool_res_nopool, train_losses_res_nopool, train_losss_res_nopool, test_losss_res_nopool, train_cons_res_nopool, test_cons_res_nopool = train_test(epochs, net_res_nopool, device, train_loader, test_loader, optimizer_res_nopool, criterion_RMSE)
net_nopool_nopool = net_res_nopool_res_nopool.cpu()
torch.save(net_nopool_nopool,'net_noadap_RMSE')
np.savez('net_res_noadap_RMSE.npz', netd=net_res_nopool_res_nopool, trainlossd=train_losses_res_nopool, trainlosssd=train_losss_res_nopool, testlosssd=test_losss_res_nopool, traincond=train_cons_res_nopool, testcond=test_cons_res_nopool)

del net_res_nopool_res_nopool
del net_nopool_nopool
del net_res_nopool

#%% no adap extra linear
hidden_channels_noadaplin = [16,32,64,128,128]
net_noadaplin = p3DNets.ResNet18_3D_noadaptive_extralin(in_channels, hidden_channels_noadaplin)
optimizer_noadaplin = torch.optim.Adam(net_noadaplin.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadaplin, (1, 96, 160, 160), device='cpu') 
#%%
net_noadaplin.apply(reset_weights)
net_noadaplin_noadaplin, train_losses_noadaplin, train_losss_noadaplin, test_losss_noadaplin, train_cons_noadaplin, test_cons_noadaplin = train_test(epochs, net_noadaplin, device, train_loader, test_loader, optimizer_noadaplin, criterion)
net_noadaplin_noadaplin = net_noadaplin_noadaplin.cpu()
torch.save(net_noadaplin_noadaplin,'net_noadaplin')
np.savez('net_noadaplin_3D.npz', trainlossd=train_losses_noadaplin, trainlosssd=train_losss_noadaplin, testlosssd=test_losss_noadaplin, traincond=train_cons_noadaplin, testcond=test_cons_noadaplin)

del net_noadaplin_noadaplin
del net_noadaplin

#%% no adap nopool
hidden_channels_noadapnopool0 = [16,32,64,128]
net_noadapnopool0 = p3DNets.ResNet18_3D_noadaptive_nopool(in_channels, hidden_channels_noadapnopool0)
optimizer_noadapnopool0 = torch.optim.Adam(net_noadapnopool0.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_noadapnopool0, (1, 96, 160, 160), device='cpu') 
#%%
net_noadapnopool0.apply(reset_weights)
net_noadapnopool0_noadapnopool0, train_losses_noadapnopool0, train_losss_noadapnopool0, test_losss_noadapnopool0, train_cons_noadapnopool0, test_cons_noadapnopool0 = train_test(epochs, net_noadapnopool0, device, train_loader, test_loader, optimizer_noadapnopool0, criterion)
net_noadapnopool0_noadapnopool0 = net_noadapnopool0_noadapnopool0.cpu()
torch.save(net_noadapnopool0_noadapnopool0,'net_noadapnopool0')
np.savez('net_noadapnopool0_3D.npz', trainlossd=train_losses_noadapnopool0, trainlosssd=train_losss_noadapnopool0, testlosssd=test_losss_noadapnopool0, traincond=train_cons_noadapnopool0, testcond=test_cons_noadapnopool0)

"""



"""
#%% smaller resnet, no adaptive
hidden_channels_res_small = [8,16,32,64,128]
net_res_small = p3DNets.ResNet18_3D_noadaptive(in_channels, hidden_channels_res_small)
optimizer_res_small = torch.optim.Adam(net_res_small.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_small, (1, 160, 160, 160), device='cpu') 
#%%
net_res_small.apply(reset_weights)
net_res_small_res_small, train_losses_res_small, train_losss_res_small, test_losss_res_small, train_cons_res_small, test_cons_res_small = train_test(epochs, net_res_small, device, train_loader, test_loader, optimizer_res_small, criterion)
np.savez('net_res_small_3D.npz', netd=net_res_small_res_small, trainlossd=train_losses_res_small, trainlosssd=train_losss_res_small, testlosssd=test_losss_res_small, traincond=train_cons_res_small, testcond=test_cons_res_small)

#%% larger resnet
hidden_channels_res_large = [32,64,128,256,128]
net_res_large = p3DNets.ResNet18_3D_noadaptive(in_channels, hidden_channels_res_large)
optimizer_res_large = torch.optim.Adam(net_res_large.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_large, (1, 96, 160, 160), device='cpu') 
#%%
net_res_large.apply(reset_weights)
net_res_large_res_large, train_losses_res_large, train_losss_res_large, test_losss_res_large, train_cons_res_large, test_cons_res_large = train_test(epochs, net_res_large, device, train_loader, test_loader, optimizer_res_large, criterion)
np.savez('net_res_large_3D.npz', netd=net_res_large_res_large, trainlossd=train_losses_res_large, trainlosssd=train_losss_res_large, testlosssd=test_losss_res_large, traincond=train_cons_res_large, testcond=test_cons_res_large)

#%% deeper resnet
hidden_channels_res_deep = [16,32,64,128]
net_res_deep = p3DNets.ResNet18_3D_d_more(in_channels, hidden_channels_res_deep)
optimizer_res_deep = torch.optim.Adam(net_res_deep.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_deep, (1, 160, 160, 160), device='cpu') 
#%%
net_res_deep.apply(reset_weights)
net_res_deep_res_deep, train_losses_res_deep, train_losss_res_deep, test_losss_res_deep, train_cons_res_deep, test_cons_res_deep = train_test(epochs, net_res_deep, device, train_loader, test_loader, optimizer_res_deep, criterion)
np.savez('net_res_deep_3D.npz', netd=net_res_deep_res_deep, trainlossd=train_losses_res_deep, trainlosssd=train_losss_res_deep, testlosssd=test_losss_res_deep, traincond=train_cons_res_deep, testcond=test_cons_res_deep)

#%% shallower resnet
hidden_channels_res_shallow = [32,64,128]
net_res_shallow = p3DNets.ResNet18_3D_d_less(in_channels, hidden_channels_res_shallow)
optimizer_res_shallow = torch.optim.Adam(net_res_shallow.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_shallow, (1, 160, 160, 160), device='cpu') 
#%%
net_res_shallow.apply(reset_weights)
net_res_shallow_res_shallow, train_losses_res_shallow, train_losss_res_shallow, test_losss_res_shallow, train_cons_res_shallow, test_cons_res_shallow = train_test(epochs, net_res_shallow, device, train_loader, test_loader, optimizer_res_shallow, criterion)
np.savez('net_res_shallow_3D.npz', netd=net_res_shallow_res_shallow, trainlossd=train_losses_res_shallow, trainlosssd=train_losss_res_shallow, testlosssd=test_losss_res_shallow, traincond=train_cons_res_shallow, testcond=test_cons_res_shallow)

#%% skip connections
net_res_noskip = p3DNets.ResNet18_3D_no_skip(in_channels, hidden_channels_res)
optimizer_res_noskip = torch.optim.Adam(net_res_noskip.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_noskip, (1, 160, 160, 160), device='cpu') 
#%%
net_res_noskip.apply(reset_weights)
net_res_noskip_res_noskip, train_losses_res_noskip, train_losss_res_noskip, test_losss_res_noskip, train_cons_res_noskip, test_cons_res_noskip = train_test(epochs, net_res_noskip, device, train_loader, test_loader, optimizer_res_noskip, criterion)
np.savez('net_res_noskip_3D.npz', netd=net_res_noskip_res_noskip, trainlossd=train_losses_res_noskip, trainlosssd=train_losss_res_noskip, testlosssd=test_losss_res_noskip, traincond=train_cons_res_noskip, testcond=test_cons_res_noskip)

#%% geen layer0 resnet
net_res_layer0 = p3DNets.ResNet18_3D_no_pool(in_channels, hidden_channels_res)
optimizer_res_layer0 = torch.optim.Adam(net_res_layer0.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_layer0, (1, 160, 160, 160), device='cpu') 
#%%
net_res_layer0.apply(reset_weights)
net_res_layer0_res_layer0, train_losses_res_layer0, train_losss_res_layer0, test_losss_res_layer0, train_cons_res_layer0, test_cons_res_layer0 = train_test(epochs, net_res_layer0, device, train_loader, test_loader, optimizer_res_layer0, criterion)
np.savez('net_res_layer0_3D.npz', netd=net_res_layer0_res_layer0, trainlossd=train_losses_res_layer0, trainlosssd=train_losss_res_layer0, testlosssd=test_losss_res_layer0, traincond=train_cons_res_layer0, testcond=test_cons_res_layer0)

#%% no batch norm
net_res_nobatchnorm = p3DNets.ResNet18_3D_no_batchnorm(in_channels, hidden_channels_res)
optimizer_res_nobatchnorm = torch.optim.Adam(net_res_nobatchnorm.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_nobatchnorm, (1, 96, 160, 160), device='cpu') 
#%%
net_res_nobatchnorm.apply(reset_weights)
net_res_nobatchnorm_res_nobatchnorm, train_losses_res_nobatchnorm, train_losss_res_nobatchnorm, test_losss_res_nobatchnorm, train_cons_res_nobatchnorm, test_cons_res_nobatchnorm = train_test(epochs, net_res_nobatchnorm, device, train_loader, test_loader, optimizer_res_nobatchnorm, criterion)
np.savez('net_res_nobatchnorm_3D.npz', netd=net_res_nobatchnorm_res_nobatchnorm, trainlossd=train_losses_res_nobatchnorm, trainlosssd=train_losss_res_nobatchnorm, testlosssd=test_losss_res_nobatchnorm, traincond=train_cons_res_nobatchnorm, testcond=test_cons_res_nobatchnorm)

#%% 320
net_res_ongescaled = p3DNets.ResNet18_3D_d(in_channels, hidden_channels_res)
optimizer_res_ongescaled = torch.optim.Adam(net_res_ongescaled.parameters(), lr=learning_rate, weight_decay=0.5)
summary(net_res_ongescaled, (1, 160, 160, 160), device='cpu') 
#%%
net_res_ongescaled.apply(reset_weights)
net_res_ongescaled_res_ongescaled, train_losses_res_ongescaled, train_losss_res_ongescaled, test_losss_res_ongescaled, train_cons_res_ongescaled, test_cons_res_ongescaled = train_test(epochs, net_res_ongescaled, device, train_loader, test_loader, optimizer_res_ongescaled, criterion)
np.savez('net_res_ongescaled_3D.npz', netd=net_res_ongescaled_res_ongescaled, trainlossd=train_losses_res_ongescaled, trainlosssd=train_losss_res_ongescaled, testlosssd=test_losss_res_ongescaled, traincond=train_cons_res_ongescaled, testcond=test_cons_res_ongescaled)
"""

#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.5)

#net.apply(reset_weights)
#net, train_losses, train_losss, test_losss, train_cons, test_cons = train_test(epochs, net, device, train_loader, test_loader, optimizer, criterion)

#%%
"""
# Plot training curves
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.xlabel('Iterations')
plt.ylabel('Loss')
xxx = torch.stack(train_losses).cpu().detach().numpy()
plt.plot(xxx)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losss, label = 'train')
plt.plot(test_losss, label = 'test')
plt.legend()
plt.grid()
"""

#%%
"""
def kfold_cv():
    # K-fold Cross Validation model evaluation
    torch.manual_seed(10)
    k_folds = 10
    kfold = KFold(n_splits=k_folds, shuffle=True)
    dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
      
    # For fold results
    results = {}
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                          dataset, 
                          batch_size=8, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                          dataset,
                          batch_size=40, sampler=test_subsampler)
               
        net.train()
        net.to(device)
        net.apply(reset_weights)
                 
        # Run the training loop for defined number of epochs
        for epoch in range(0, epochs):
              
            # Print epoch
            print(f'Starting epoch {epoch+1}')
                
            # Iterate over the DataLoader for training data
            for i, (x_batch, y_batch) in enumerate(trainloader, 0):
              
              # Set to same device
              x_batch, y_batch = x_batch.to(device), y_batch.to(device)
              
              # Set the gradients to zero
              optimizer.zero_grad()
              
                        #outputs = network(inputs)
              # Perform forward pass
              y_pred = net(x_batch)
              
              # Compute the loss
              loss = criterion(y_pred, y_batch)
              train_losses.append(loss)
              
              # Perform backward pass
              loss.backward()
              
              # Perform optimization
              optimizer.step() 
                                      
        # Process is complete.
        print('Training process has finished. Saving trained model.')
        
        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(net.state_dict(), save_path)
        
        # Print about testing
        print('Starting testing')
            
        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
    
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
        
                # Get inputs
                inputs, targets = data
          
                # Generate outputs
                outputs = net(inputs)
          
                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
              
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
      print(f'Fold {key}: {value} %')
      sum += value
    print(f'Average: {sum/len(results.items())} %')
    return sum/len(results.items())
"""


