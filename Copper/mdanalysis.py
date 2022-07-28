# mdanalysis example
import MDAnalysis as mda
import maicos
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np

#%% functions below were all used to split larger dump file into multiple smaller ones
def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number, line.rstrip()))
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results

def listToString(s): 

    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 

with open('dump.03.lammpstrj','r') as fin:
    list_of_occurences = search_string_in_file("dump.03.lammpstrj",'ITEM: TIMESTEP')
    infile = fin.read().splitlines(True)
    beg = 0
    for i in np.arange(1,7):
                
        outfilename = "out."+str(i)+".lammpstrj"
        outfile = open(outfilename, 'w')

        index = int(i*len(list_of_occurences)/6)
        end = list_of_occurences[index-1][0]
        if i == 6:
            end = 0
        file = listToString(infile[beg:end-1])
        outfile.write(file)
        beg = end - 1

#%% load resulting small dump files
u_1 = mda.Universe("out.1.lammpstrj", topology_format='LAMMPSDUMP',lengthunit="A", timeunit="ps")
u_2 = mda.Universe("out.2.lammpstrj", topology_format='LAMMPSDUMP',lengthunit="A", timeunit="ps")
u_3 = mda.Universe("out.3.lammpstrj", topology_format='LAMMPSDUMP',lengthunit="A", timeunit="ps")
u_4 = mda.Universe("out.4.lammpstrj", topology_format='LAMMPSDUMP',lengthunit="A", timeunit="ps")
u_5 = mda.Universe("out.5.lammpstrj", topology_format='LAMMPSDUMP',lengthunit="A", timeunit="ps")
u_6 = mda.Universe("out.6.lammpstrj", topology_format='LAMMPSDUMP',lengthunit="A", timeunit="ps")

#%%
color = ['r','b','g','c','y','k']

for i in np.arange(1,7):
    if i == 1:
        u = u_1
    if i == 2:
        u = u_2
    if i == 3:
        u = u_3
    if i == 4:
        u = u_4
    if i == 5:
        u = u_5
    if i == 6:
        u = u_6

    grp_h2o = u.select_atoms('type 1')
    dplanh2o = maicos.density_planar(grp_h2o,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
    dplanh2o.run()
    zcoorh2o = dplanh2o.results['z']
    densh2o = dplanh2o.results['dens_mean']*18.015
    plt.figure(1)
    plt.grid()
    plt.title('Water')
    plt.plot(zcoorh2o,densh2o,color[i-1])
    cminth2o = integrate.cumulative_trapezoid(densh2o[:,0],zcoorh2o)*5/18.015
    plt.plot(zcoorh2o[1:],cminth2o,color[i-1])
    
    grp_co2 = u.select_atoms('type 3')
    dplanco2 = maicos.density_planar(grp_co2,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
    dplanco2.run()
    zcoorco2 = dplanco2.results['z']
    densco2 = dplanco2.results['dens_mean']*44.01
    plt.figure(2)
    plt.grid()
    plt.title('Koolstofdioxide')
    plt.plot(zcoorco2,densco2,color[i-1])
    cmintco2 = integrate.cumulative_trapezoid(densco2[:,0],zcoorco2)*50/44.01
    plt.plot(zcoorco2[1:],cmintco2,color[i-1])
    
    grp_hco3 = u.select_atoms('type 5')
    dplanhco3 = maicos.density_planar(grp_hco3,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
    dplanhco3.run()
    zcoorhco3 = dplanhco3.results['z']
    denshco3 = dplanhco3.results['dens_mean']*61.01
    plt.figure(3)
    plt.grid()
    plt.title('Bicarbonaat')
    plt.plot(zcoorhco3,denshco3,color[i-1])
    cminthco3 = integrate.cumulative_trapezoid(denshco3[:,0],zcoorhco3)*50/61.01
    plt.plot(zcoorhco3[1:],cminthco3,color[i-1])
    
    grp_k = u.select_atoms('type 10')
    dplank = maicos.density_planar(grp_k,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
    dplank.run()
    zcoork = dplank.results['z']
    densk = dplank.results['dens_mean']*39.098
    plt.figure(4)
    plt.grid()
    plt.title('Kalium')
    plt.plot(zcoork,densk,color[i-1])
    cmintk = integrate.cumulative_trapezoid(densk[:,0],zcoork)*50/39.098
    plt.plot(zcoork[1:],cmintk,color[i-1])
   
    
for i in np.arange(1,5):
    plt.figure(i)
    plt.legend(['Eerste 1/6','Eerste 1/6','2/6','2/6','3/6','3/6','4/6','4/6','5/6','5/6','Laatste 1/6','Laatste 1/6'])