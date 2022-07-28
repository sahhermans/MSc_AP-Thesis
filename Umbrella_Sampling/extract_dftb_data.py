import pandas as pd

with open('detailed.out', 'r') as fin:
    datadftb = fin.read().splitlines(True)
file = datadftb[14:398]
with open('charges.txt', 'w') as fout:
    fout.writelines(file)
infiledftb = pd.read_csv("charges.txt",sep = '      ',header =None,engine = 'python')
infiledftb.columns = ['Atom no.','q']
  
with open('data.02.lammps', 'r') as fin:
    datalammps = fin.read().splitlines(True)
beginning = datalammps[:32]
ending = datalammps[7090:]

exclude = list(range(0, 30000))
del exclude[33:7090]
infilelammps = pd.read_csv("data.02.lammps", skiprows = exclude, sep = ' ', header = None)
infilelammps.columns = ['Atom no.','Molecule no.','Type','q','x','y','z','nx','ny','nz']
                                       
infilelammps = infilelammps.sort_values('Atom no.')
infilelammpsCu = infilelammps[(infilelammps['Type'] == 10)]
infilelammpsRest = infilelammps[(infilelammps['Type'] != 10)]
infilelammpsCu = infilelammpsCu.assign(q = list(infiledftb['q']))
if list(infiledftb['q']) != list(infilelammpsCu['q']):
    print("ERROR: Copying charges unsuccessful")

infilelammpsCuAsString = infilelammpsCu.to_string(header=False,index=False,float_format='%.16e')
infilelammpsRestAsString = infilelammpsRest.to_string(header=False,index=False,float_format='%.16e')

with open('data.04.lammps', 'w') as fout:
    fout.writelines(beginning)
    fout.writelines('\n')
    fout.writelines(infilelammpsCuAsString + '\n')
    fout.writelines(infilelammpsRestAsString + '\n')
    fout.writelines('\n')
    
    fout.writelines(ending)