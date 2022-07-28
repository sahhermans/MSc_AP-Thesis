with open('data.02.lammps', 'r') as fin:
    datalammps = fin.read().splitlines(True)
beginning = datalammps[:11]
ending = datalammps[12:]

with open('data.02.lammps', 'w') as fout:
    fout.writelines(beginning)
    fout.writelines('-30.0000 30.0000 zlo zhi')
    fout.writelines('\n')
    fout.writelines(ending)