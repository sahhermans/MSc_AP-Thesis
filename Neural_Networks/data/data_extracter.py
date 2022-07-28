# setup
import numpy as np
import pandas as pd
import h5py

#%%

list = ['00004000','40008000','800012000','1200016000','1600020000','2000024000',
        '2400028000','lammps15','lammps21','lammps30','stootdftb','stootlammps','wissellammps']

#%%
K = np.empty(shape=(37,3,1000,13))
Cl = np.empty(shape=(37,3,1000,13))
O2 = np.empty(shape=(2009,3,1000,13))
H2 = np.empty(shape=(2009*2,3,1000,13))
X = np.empty(shape=(37+37,3,1000,13))
X_q = np.empty(shape=(37+37,4,1000,13))
X_H2O = np.empty(shape=(37+37+50*3,3,1000,13))
q_2D = np.empty(shape=(128,4,1000,13))
q = np.empty(shape=(128,1000,13))
#input_space = np.zeros(shape=(160,160,160,1000))

for j in range(len(list)):    
    for i in range(0,1000): 
            
        df = pd.DataFrame(np.array(h5py.File(list[j] + '.h5')['data'][i]))
        df.columns = ['Atom no.','Molecule no.','Type','q','x','y','z']
        df = df.sort_values('Atom no.')
            
        xyz = df[(df['Type'] != 5)]
        xyz = xyz[xyz['Type'] != 6]
           
        xyz_O2 = xyz[xyz['Type'] == 1]
        xyz_H2 = xyz[xyz['Type'] == 2]
        
        xyz_O2_H2O = xyz_O2.copy()
        xyz_H2_H2O = xyz_H2.copy()
        
        xyz_O2_H2O = xyz_O2_H2O.sort_values('z')[:50]
        mol_num = xyz_O2_H2O['Molecule no.'].to_numpy()
        xyz_H2_H2O = xyz_H2[xyz_H2['Molecule no.'] == mol_num[0]]
        for k in range(1,50):
            xyz_H2_H2O = pd.concat([xyz_H2_H2O,xyz_H2[xyz_H2['Molecule no.'] == mol_num[k]]],axis=0)
        
        xyz_H2O2_H2O = pd.concat([xyz_O2_H2O,xyz_H2_H2O],axis=0)
        xyz_H2O2_H2O = xyz_H2O2_H2O.sort_values('Atom no.')
        
        xyz = xyz[xyz['Type'] != 1]
        xyz = xyz[xyz['Type'] != 2]
           
        df = df[(df['Type'] == 5)]
        df = df[df['z'] == max(df['z'])]
        
        xyz_K = xyz[xyz['Type'] == 4]
        xyz_Cl = xyz[xyz['Type'] == 3]
                
        Cl[:,:,i,j] = xyz_Cl[['x','y','z']]
        K[:,:,i,j]  = xyz_K[['x','y','z']]
        
        O2[:,:,i,j] = xyz_O2[['x','y','z']]
        H2[:,:,i,j]  = xyz_H2[['x','y','z']]
        
        X[:,:,i,j] = xyz[['x','y','z']]
        X_q[:,:,i,j] = xyz[['q','x','y','z']]
        X_H2O[:,:,i,j] = pd.concat([xyz[['x','y','z']],xyz_H2O2_H2O[['x','y','z']]],axis=0)
        
        q_2D[:,:,i,j] = df[['q','x','y','z']] 
        q[:,i,j] = df['q'] 
 
#%% train params
np.save('K_tensor',K)
np.save('Cl_tensor',Cl)
np.save('O2_tensor',O2)
np.save('H2_tensor',H2)
np.save('X_tensor',X)
np.save('X_q_tensor',X_q)
np.save('X_H2O_tensor',X_H2O)
np.save('q_unet_tensor',q_2D)
np.save('q_tensor',q)

#%%
K = np.load('K_tensor.npy')
Cl = np.load('Cl_tensor.npy')
O2 = np.load('O2_tensor.npy')
H2 = np.load('H2_tensor.npy')
X = np.load('X_tensor.npy')
X_q = np.load('X_q_tensor.npy')
X_H2O = np.load('X_H2O_tensor.npy')

#%%
q_2D = np.load('q_unet_tensor.npy')
q = np.load('q_tensor.npy')

#%%
N = 500
list = [1]
for i in range(len(list)):
    for j in range(N):
        ind = np.lexsort((q_2D[:,1,j,i],q_2D[:,2,j,i]))
        q_2D[:,:,j,i] = q_2D[ind,:,j,i] 

np.save('q_unet_tensor_sorted',q_2D)

#%%
K_copy = K.copy()
Cl_copy = Cl.copy()
O2_copy = O2.copy()
H2_copy = H2.copy()
X_copy = X.copy()
X_q_copy = X_q.copy()
X_H2O_copy = X_H2O.copy()
q_2D_copy = q_2D.copy()
q_copy = q.copy()

N = 500
list = [1]

def rotate(input, q_2D_copy):
    # x = -x, y = -y
    X_r = input.copy()
    q_2D_r = q_2D_copy.copy()

    x_min = np.min(q_2D_r[:,1,:,:])
    y_min = np.min(q_2D_r[:,2,:,:])
    
    box_length = -x_min*2
    translate = -x_min/16
    
    X_r[:,0,:,:] = X_r[:,0,:,:] + translate
    X_r[:,1,:,:] = X_r[:,1,:,:] + translate
    X_r_copy = X_r.copy()
    X_r[:,0,:,:] = -X_r_copy[:,0,:,:]
    X_r[:,1,:,:] = -X_r_copy[:,1,:,:]
    X_r[:,0,:,:] = X_r[:,0,:,:] - translate
    X_r[:,1,:,:] = X_r[:,1,:,:] - translate
    
    q_2D_r[:,1,:,:] = q_2D_r[:,1,:,:] + translate
    q_2D_r[:,2,:,:] = q_2D_r[:,2,:,:] + translate
    q_2D_r_copy = q_2D_r.copy()
    q_2D_r[:,1,:,:] = -q_2D_r_copy[:,1,:,:]
    q_2D_r[:,2,:,:] = -q_2D_r_copy[:,2,:,:]
    q_2D_r[:,1,:,:] = q_2D_r[:,1,:,:] - translate
    q_2D_r[:,2,:,:] = q_2D_r[:,2,:,:] - translate
    
    for i in range(len(list)):
        for j in range(N):
            for k in range(len(input[:,0,0,0])):
                if X_r[k,0,j,i] < x_min:
                    X_r[k,0,j,i] += box_length
                if X_r[k,1,j,i] < y_min:
                    X_r[k,1,j,i] += box_length
    
    for i in range(len(list)):
        for j in range(N):
            ind = np.lexsort((q_2D_r[:,1,j,i],q_2D_r[:,2,j,i]))
            q_2D_r[:,:,j,i] = q_2D_r[ind,:,j,i] 
            
    return X_r, q_2D_r

def mirrorxy(input, q_2D_copy):
    # along x = y
    # x = y, y = x
    X_xy = input.copy()
    q_2D_xy = q_2D_copy.copy()
    
    x_min = np.min(q_2D_xy[:,1,:,:])
    y_min = np.min(q_2D_xy[:,2,:,:])
    
    box_length = -x_min*2
    translate = -x_min/16
    
    X_xy[:,0,:,:] = X_xy[:,0,:,:] + translate
    X_xy[:,1,:,:] = X_xy[:,1,:,:] + translate
    X_xy_copy = X_xy.copy()
    X_xy[:,0,:,:] = X_xy_copy[:,1,:,:]
    X_xy[:,1,:,:] = X_xy_copy[:,0,:,:]
    X_xy[:,0,:,:] = X_xy[:,0,:,:] - translate
    X_xy[:,1,:,:] = X_xy[:,1,:,:] - translate
    
    q_2D_xy[:,1,:,:] = q_2D_xy[:,1,:,:] + translate
    q_2D_xy[:,2,:,:] = q_2D_xy[:,2,:,:] + translate
    q_2D_xy_copy = q_2D_xy.copy()
    q_2D_xy[:,1,:,:] = q_2D_xy_copy[:,2,:,:]
    q_2D_xy[:,2,:,:] = q_2D_xy_copy[:,1,:,:]
    q_2D_xy[:,1,:,:] = q_2D_xy[:,1,:,:] - translate
    q_2D_xy[:,2,:,:] = q_2D_xy[:,2,:,:] - translate
    
    for i in range(len(list)):
        for j in range(N):
            for k in range(len(input[:,0,0,0])):
                if X_xy[k,0,j,i] < x_min:
                    X_xy[k,0,j,i] += box_length
                if X_xy[k,1,j,i] < y_min:
                    X_xy[k,1,j,i] += box_length
        
    for i in range(len(list)):
        for j in range(N):
            ind = np.lexsort((q_2D_xy[:,1,j,i],q_2D_xy[:,2,j,i]))
            q_2D_xy[:,:,j,i] = q_2D_xy[ind,:,j,i] 
            
    return X_xy, q_2D_xy
                
def mirrorminxy(input, q_2D_copy):
    # along x = -y
    # x = -y, y = -x
    X_minxy = input.copy()
    q_2D_minxy = q_2D_copy.copy()
    
    x_min = np.min(q_2D_minxy[:,1,:,:])
    y_min = np.min(q_2D_minxy[:,2,:,:])
    
    box_length = -x_min*2
    translate = -x_min/16
    
    X_minxy[:,0,:,:] = X_minxy[:,0,:,:] + translate
    X_minxy[:,1,:,:] = X_minxy[:,1,:,:] + translate
    X_minxy_copy = X_minxy.copy()
    X_minxy[:,0,:,:] = -X_minxy_copy [:,1,:,:]
    X_minxy[:,1,:,:] = -X_minxy_copy [:,0,:,:]
    X_minxy[:,0,:,:] = X_minxy[:,0,:,:] - translate
    X_minxy[:,1,:,:] = X_minxy[:,1,:,:] - translate
    
    q_2D_minxy[:,1,:,:] = q_2D_minxy[:,1,:,:] + translate
    q_2D_minxy[:,2,:,:] = q_2D_minxy[:,2,:,:] + translate
    q_2D_minxy_copy = q_2D_minxy.copy()
    q_2D_minxy[:,1,:,:] = -q_2D_minxy_copy [:,2,:,:]
    q_2D_minxy[:,2,:,:] = -q_2D_minxy_copy [:,1,:,:]
    q_2D_minxy[:,1,:,:] = q_2D_minxy[:,1,:,:] - translate
    q_2D_minxy[:,2,:,:] = q_2D_minxy[:,2,:,:] - translate
    
    for i in range(len(list)):
        for j in range(N):
            for k in range(len(input[:,0,0,0])):
                if X_minxy[k,0,j,i] < x_min:
                    X_minxy[k,0,j,i] += box_length
                if X_minxy[k,1,j,i] < y_min:
                    X_minxy[k,1,j,i] += box_length

    for i in range(len(list)):
        for j in range(N):
            ind = np.lexsort((q_2D_minxy[:,1,j,i],q_2D_minxy[:,2,j,i]))
            q_2D_minxy[:,:,j,i] = q_2D_minxy[ind,:,j,i]                
            
    return X_minxy, q_2D_minxy

#%%
X_r, q_2D_r = rotate(X_copy,q_2D_copy)
X_xy, q_2D_xy = mirrorxy(X_copy,q_2D_copy)
X_minxy, q_2D_minxy = mirrorminxy(X_copy,q_2D_copy)

X_H2O_r, q_2D_r = rotate(X_H2O_copy,q_2D_copy)
X_H2O_xy, q_2D_xy = mirrorxy(X_H2O_copy,q_2D_copy)
X_H2O_minxy, q_2D_minxy = mirrorminxy(X_H2O_copy,q_2D_copy)

Cl_r, q_2D_r = rotate(Cl_copy,q_2D_copy)
Cl_xy, q_2D_xy = mirrorxy(Cl_copy,q_2D_copy)
Cl_minxy, q_2D_minxy = mirrorminxy(Cl_copy,q_2D_copy)

K_r, q_2D_r = rotate(K_copy,q_2D_copy)
K_xy, q_2D_xy = mirrorxy(K_copy,q_2D_copy)
K_minxy, q_2D_minxy = mirrorminxy(K_copy,q_2D_copy)

O2_r, q_2D_r = rotate(O2_copy,q_2D_copy)
O2_xy, q_2D_xy = mirrorxy(O2_copy,q_2D_copy)
O2_minxy, q_2D_minxy = mirrorminxy(O2_copy,q_2D_copy)

H2_r, q_2D_r = rotate(H2_copy,q_2D_copy)
H2_xy, q_2D_xy = mirrorxy(H2_copy,q_2D_copy)
H2_minxy, q_2D_minxy = mirrorminxy(H2_copy,q_2D_copy)

#%%
np.save('K_tensor_r',K_r)
np.save('Cl_tensor_r',Cl_r)
np.save('O2_tensor_r',O2_r)
np.save('H2_tensor_r',H2_r)
np.save('X_tensor_r',X_r)
np.save('X_H2O_tensor_r',X_H2O_r)
np.save('q_tensor_r',q_2D_r)

np.save('K_tensor_xy',K_xy)
np.save('Cl_tensor_xy',Cl_xy)
np.save('O2_tensor_xy',O2_xy)
np.save('H2_tensor_xy',H2_xy)
np.save('X_tensor_xy',X_xy)
np.save('X_H2O_tensor_xy',X_H2O_xy)
np.save('q_tensor_xy',q_2D_xy)

np.save('K_tensor_minxy',K_minxy)
np.save('Cl_tensor_minxy',Cl_minxy)
np.save('O2_tensor_minxy',O2_minxy)
np.save('H2_tensor_minxy',H2_minxy)
np.save('X_tensor_minxy',X_minxy)
np.save('X_H2O_tensor_minxy',X_H2O_minxy)
np.save('q_tensor_minxy',q_2D)
