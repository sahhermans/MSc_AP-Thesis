# script that can be used to shrink simulation box (for more efficient simulation) when moving from input.02 to input.03 or further
with open('data.02.lammps', 'r') as fin:
    datalammps = fin.read().splitlines(True)
beginning = datalammps[:15]
ending = datalammps[16:]

with open('data.02.lammps', 'w') as fout:
    fout.writelines(beginning)
    fout.writelines('-43.0000 43.0000 zlo zhi')
    fout.writelines('\n')
    fout.writelines(ending)

