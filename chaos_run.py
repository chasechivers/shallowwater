from ShallowWater import ShallowWater as Model
from utility_funcs import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


# Define shape and temperature of the bottom boundary condition
def plume_shape_function(x, w, a, c, b=2, p=24):
	return a * erf(b * (x / w) ** p) + -1e-6 * x + c


# ice shell properties
D = 5e3  # m, brittle ice shell thickness
Tsurf = 110.  # K, surface temperature
Tbot = 260.  # K, bottom boundary temperature

# melt lens geometry
d = 3e3  # m, depth from surface
R = 50e3  # m, melt lens radius
Rr = (1 - 0.2) * R  # m, short (top) radius of melt lens geometry
h = 2 * (D - d)  # m, melt lens thickness
w = 2.5 * R  # m, set domain width

# salinity choices
comp = "NaCl"  # dominate oceanic salt
conc = 35.  # g/kg, concentration

# set spatial step sizes
dx = R / int(2.4e3 / 20)  # m, horizontal step size, scaled to 1 km depth sill
dz = 20  # m, vertical step size

# set boundary conditions
topBC = True  # constant surface temperature
sidesBC = "NoFlux"
bottomBC = "GhostIce"  #

# output choices
tmp_dir = './tmp/'  # directory to save temporary files
out_dir = './results/'  # directory to save final results/model file
name = f'meltlens_{comp}'  # unique identifier
output_freq = 1000 * 3.154e7  # output every 100 years
simulation_time = 350e3 * 3.154e7  # run simulation for 350 thousand years

md = Model(w=w, D=D, dx=dx, dz=dz, verbose=True)

md.init_T(Tsurf=Tsurf, Tbot=Tbot)
md.init_intrusion(depth=d, thickness=h, radius=R, geometry="chaos", inner_radius=Rr)
md.init_salinity(composition=comp, concentration=conc, T_match=False, in_situ=True)

TBotGhost = plume_shape_function(md.X[0, :], md.w,
                                 md.Tsurf * (md.Tbot / md.Tsurf) ** (1 + 0.5 * md.dz / md.D) - md.Tm[-1, -1],
                                 md.Tm[-1, -1])

md.set_boundaryconditions(sides=sidesBC,
                          top=topBC,
                          bottom=bottomBC, TBotGhost=TBotGhost)

md.set_outputs(output_frequency=int(output_freq / md.dt), tmp_dir=tmp_dir, tmp_file_name=name)
md.solve_heat(final_time=simulation_time)

# Gather results outputs to save them
rs = md.outputs.get_all_data(del_files=True)

print('Saving model')
md.write()
print('Saving results')
save_data(rs,
          'rs_{}'.format(md.outputs.tmp_data_file_name.split('tmp_data_')[1]),
          out_dir)

# Plot the x = dx slice of the temperature evolution over time
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
