import pandas as pd
import numpy as np

with open('data.lammps', 'r') as fin:
    datalammps = fin.read().splitlines(True)
beginning = datalammps[:13]

exclude = list(range(0, 30000))
del exclude[14:350]
infilelammps = pd.read_csv("data.lammps", skiprows = exclude, sep = ' ', header = None)
infilelammps.columns = ['A','M','T','q','x','y','z']
#infilelammps = infilelammps.drop('random',1)

array = np.zeros(336)
array = [int(x) for x in array]
infilelammps = infilelammps.assign(M = list(array))

infilelammps2 = infilelammps.copy()
infilelammps2['x'] = infilelammps2['x'] + np.cos(np.deg2rad(30))*1.427 - 12*np.cos(np.deg2rad(30))*1.427
infilelammps2['y'] = infilelammps2['y'] - 21*1.427/2      
infilelammps2['z'] = infilelammps2['z'] - 83.1427

infilelammps['x'] = infilelammps['x'] + np.cos(np.deg2rad(30))*1.427 - 12*np.cos(np.deg2rad(30))*1.427
infilelammps['y'] = infilelammps['y'] - 21*1.427/2                    
infilelammps['z'] = infilelammps['z'] + 83.1427

infilelammps = pd.concat([infilelammps,infilelammps2],axis=0)

array = np.linspace(1,336*2,672)
array = [int(x) for x in array]
infilelammps = infilelammps.assign(A = list(array))

array = np.hstack([np.ones(336),np.ones(336)*2])
array = [int(x) for x in array]
infilelammps = infilelammps.assign(T = list(array))

infilelammpsAsString = infilelammps.to_string(header=False,index=False,float_format='%.7f')

with open('data.practice.lammps', 'w') as fout:
    fout.writelines(infilelammpsAsString)

with open('data.practice.lammps', 'r') as fin:
    practicedata = fin.read().splitlines(True)
for i in range(len(practicedata)):
    practicedata[i] = practicedata[i].lstrip().replace("    ", " ").replace("   ", " ").replace("  ", " ")

with open('data.test.lammps', 'w') as fout:
    fout.writelines(beginning)
    fout.writelines(practicedata)
    fout.writelines('\n')
    fout.writelines('\n')

