from ShallowWater import ShallowWater as Model
from utility_funcs import *
import matplotlib.pyplot as plt
import numpy as np

# ice shell properties
D = 5e3  # m, brittle ice shell thickness
Tsurf = 110.  # K, surface temperature
Tbot = 273.15  # K, bottom boundary temperature

# sill geometry
h = 1e3  # m, sill thickness
d = 1e3  # m, sill depth from surface
R = 2.4 * d  # m, sill radius (Michaut & Manga, 2017)
w = 2.5 * R  # m, set domain width

# salinity choices
comp = "MgSO4"  # "NaCl"  # dominate oceanic salt
conc = 12.3  # 35.  # g/kg, concentration

# set spatial step sizes
dx = 50  # m, horizontal step size
dz = 50  # m, vertical step size

# set boundary conditions
topBC = "Radiative"
sidesBC = "NoFlux"
bottomBC = True
qbot = 0

# output choices
tmp_dir = './tmp/'  # directory to save temporary files
out_dir = './results/'  # directory to save final results/model file
name = 'shallowwater_{comp}'  # unique identifier
output_freq = 10 * 3.154e7  # output every 100 years
simulation_time = 30e3 * 3.154e7  # run simulation for 200 thousand years

md = Model(w=w, D=D, dx=dx, dz=dz, coordinates=coords, verbose=True)
md.init_T(Tsurf=Tsurf, Tbot=Tbot)
md.init_intrusion(depth=d, thickness=h, radius=R)  # , geometry="sheet")
md.init_salinity(composition=comp, concentration=conc, shell=True)  # , use_interpolator=True)

md.set_boundaryconditions(top=topBC, sides=sidesBC, bottom=bottomBC)
md.set_outputs(output_frequency=int(output_freq / md.dt), tmp_dir=tmp_dir, tmp_file_name=name)
md.solve_heat(final_time=simulation_time)
# Gather results outputs and save them
rs = md.outputs.get_all_data(del_files=True)
'''
print('Saving model')
save_data(md, 'md_{}'.format(md.outputs.tmp_data_file_name.split('tmp_data_')[1]), out_dir)
print('Saving results')
save_data(rs,
		 'rs_{}'.format(md.outputs.tmp_data_file_name.split('tmp_data_')[1]),
		 out_dir)
'''
# Plot a the x = dx slice of the temperature evolution over time
plt.figure()
# Temperature evolution over time
plt.pcolormesh(rs['time'] / md.constants.styr / 1e3, md.Z[:, 1] / 1e3, rs['T'][:, :, 1].T, cmap="plasma")
plt.colorbar(label="Temperature, K")
# Plot the phase boundary over time at x = dx
plt.contour(rs['time'] / md.constants.styr / 1e3, md.Z[:, 1] / 1e3, rs['phi'][:, :, 1].T, levels=[0, 1], color='k')
plt.xlabel(r"Time after emplacement, 10$^3$ years")
plt.ylabel("Depth from surface, km")
plt.ylim(md.Z[-1, 1] / 1e3, md.Z[0, 1])
plt.show()
