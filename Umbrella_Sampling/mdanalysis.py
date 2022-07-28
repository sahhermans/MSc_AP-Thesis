# mdanalysis practice

import MDAnalysis as mda
import maicos
import matplotlib.pyplot as plt

u = mda.Universe("dump.03.lammpstrj", topology_format='LAMMPSDUMP')

#print(u)
#print(u.atoms[50:60].masses)
#print(u.atoms[:20].residues)
#print(u.atoms[-20:].segments)

#t3 = u.select_atoms('type 1')
#print(t3.positions)
#print(t3.positions.shape)
#print(t3.center_of_mass())
#print(u.bonds)



grp_h2o = u.select_atoms('type 1')
dplanh2o = maicos.density_planar(grp_h2o,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
dplanh2o.run()
zcoorh2o = dplanh2o.results['z']
densh2o = dplanh2o.results['dens_mean']*18
plt.plot(zcoorh2o,densh2o)

#ddiph2o = maicos.dipole_angle(grp_h2o,dim = 0)
#ddiph2o.run()
#timeh2o = dplanh2o.results['t']
#dipoleh2o = dplanh2o.results['cos_theta_i']
#plt.plot(timeh2o,dipoleh2o)


grp_h2o = u.select_atoms('type 1')
dplanh2o = maicos.density_planar(grp_h2o,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
dplanh2o.run()
zcoorh2o = dplanh2o.results['z']
densh2o = dplanh2o.results['dens_mean']*18.015
plt.figure(1)
plt.title('Water')
plt.plot(zcoorh2o,densh2o)

grp_co2 = u.select_atoms('type 3')
dplanco2 = maicos.density_planar(grp_co2,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
dplanco2.run()
zcoorco2 = dplanco2.results['z']
densco2 = dplanco2.results['dens_mean']*44.01
plt.figure(2)
plt.title('Koolstofdioxide')
plt.plot(zcoorco2,densco2)

grp_hco3 = u.select_atoms('type 5')
dplanhco3 = maicos.density_planar(grp_hco3,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
dplanhco3.run()
zcoorhco3 = dplanhco3.results['z']
denshco3 = dplanhco3.results['dens_mean']*61.01
plt.figure(3)
plt.title('Bicarbonaat')
plt.plot(zcoorhco3,denshco3)

grp_k = u.select_atoms('type 10')
dplank = maicos.density_planar(grp_k,binwidth=0.1)  #,binwidth=0.1,mass=18.0,dim=2) #,binwidth=0.5,dim=2,mass=6)
dplank.run()
zcoork = dplank.results['z']
densk = dplank.results['dens_mean']*39.098
plt.figure(4)
plt.title('Kalium')
plt.plot(zcoork,densk)











