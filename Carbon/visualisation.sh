# load trajectory
#topo readlammpsdata data.01.lammps
#topo guessatom lammps data
mol new {G:/carbon/-4/per/c0-1250\dumpsq.04.lammpstrj} type {lammpstrj} first 0 last -1 step 1 waitfor 1

rotate x by 90
rotate z by -90

display resize 900 900
display ambientocclusion on
display aoambient 0.75
display aodirect 0.75
material change opacity Transparent 0.15
material add Transparent2
material change ambient Transparent2 0.0
material change specular Transparent2 0.7
material change diffuse Transparent2 0.7
material change shininess Transparent2 0.15
material change opacity Transparent2 0.2

set sel [atomselect top {type 3}]
$sel set name O2
$sel set element O
$sel delete
set sel [atomselect top {type 4}]
$sel set name H2
$sel set element H
$sel delete
set sel [atomselect top {type 5}]
$sel set name Cl
$sel set element Cl
$sel delete
set sel [atomselect top {type 6}]
$sel set name K
$sel set element K
$sel delete
set sel [atomselect top {type 1}]
$sel set name C1
$sel set element C
$sel delete
set sel [atomselect top {type 2}]
$sel set name C2
$sel set element C
$sel delete
mol reanalyze 0

#mol addfile dump.03.dcd waitfor all

mol delrep 0 0
set rep 0
mol addrep 0
mol modselect ${rep} 0 {name C1 C2}
mol modstyle ${rep} 0 VDW 0.4 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {name O2 H2}
mol modstyle ${rep} 0 VDW 0.1 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {name Cl}
mol modstyle ${rep} 0 VDW 0.6 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {name K}
mol modstyle ${rep} 0 VDW 0.6 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {name O2 H2}
mol modstyle ${rep} 0 DynamicBonds 1.6 0.1 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol modcolor 2 0 ResID
mol modcolor 3 0 ColorID 1
 
mol modmaterial 2 0 Diffuse
mol modmaterial 3 0 Diffuse

mol modstyle 1 0 VDW 0.000000 12.000000
mol modstyle 4 0 DynamicBonds 1.100000 0.100000 38.000000

mol modcolor 0 0 Name
mol modmaterial 0 0 Steel
mol modstyle 0 0 VDW 1.000000 12.000000