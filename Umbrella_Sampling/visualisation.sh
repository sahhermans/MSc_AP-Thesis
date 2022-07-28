# load trajectory
#topo readlammpsdata data.01.lammps
#topo guessatom lammps data
mol new {C:/thesis/final_scaled/umbrella_sampling/KCl_Joung/dump.04.lammpstrj} type {lammpstrj} first 0 last -1 step 1 waitfor 1

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

set sel [atomselect top {type 1}]
$sel set name O2
$sel set element O
$sel delete
set sel [atomselect top {type 2}]
$sel set name H2
$sel set element H
$sel delete
set sel [atomselect top {type 3}]
$sel set name C
$sel set element C
$sel delete
set sel [atomselect top {type 4}]
$sel set name O
$sel set element O
$sel delete
set sel [atomselect top {type 5}]
$sel set name O6
$sel set element O
$sel delete
set sel [atomselect top {type 6}]
$sel set name C5
$sel set element C
$sel delete
set sel [atomselect top {type 7}]
$sel set name O4
$sel set element O
$sel delete
set sel [atomselect top {type 8}]
$sel set name O5
$sel set element O
$sel delete
set sel [atomselect top {type 9}]
$sel set name H6
$sel set element H
$sel delete
set sel [atomselect top {type 10}]
$sel set name K
$sel set element K
$sel delete
set sel [atomselect top {type 11}]
$sel set name Cu1
$sel set element Cu
$sel delete
set sel [atomselect top {type 12}]
$sel set name Cu2
$sel set element Cu
$sel delete
mol reanalyze 0

#mol addfile dump.02.dcd waitfor all

mol delrep 0 0
set rep 0
mol addrep 0
mol modselect ${rep} 0 {type 11 12}
mol modstyle ${rep} 0 VDW 0.4 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {type 1 2}
mol modstyle ${rep} 0 VDW 0.1 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {type 3 4}
mol modstyle ${rep} 0 VDW 0.6 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {type 5 6 7 8 9}
mol modstyle ${rep} 0 VDW 0.6 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {type 10}
mol modstyle ${rep} 0 VDW 0.6 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {type 1 2}
mol modstyle ${rep} 0 DynamicBonds 1.6 0.1 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {type 3 4}
mol modstyle ${rep} 0 DynamicBonds 1.6 0.3 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

mol addrep 0
incr rep
mol modselect ${rep} 0 {type 5 6 7 8 9}
mol modstyle ${rep} 0 DynamicBonds 1.6 0.3 12.0
mol modmaterial ${rep} 0 Opaque
mol modcolor 0 0 Element

translate by 0 0.1 0
scale by 4
