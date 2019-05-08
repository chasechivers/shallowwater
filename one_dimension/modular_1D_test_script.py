from one_dimension.modular_1D import IceSystem
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='colorblind', style='whitegrid')

# Test Stefan Problem
Lz = 1
dz = 0.005
dt = 0.1
ice = IceSystem(Lz, dz, dt)
Ttop = T1 = 110
Tbot = T0 = 273.15
ice.init_T(Ttop, Tbot)
ice.T[1:] = 273.15
ice.phi[:] = 1.
ice.constants.rho_w = ice.constants.rho_i
ice.constants.cp_w = ice.constants.cp_i
ice.constants.kw = ice.constants.ki
ice.init_vol_avgs(kT='non', cpT='non')
T_i, phi_i, k_i = ice.T.copy(), ice.phi.copy(), ice.k.copy()
ice.set_BC()

# determine time needed to freeze whole domain
kappa = ice.constants.ki / (ice.constants.cp_i * ice.constants.rho_i)
Stf = ice.constants.cp_i * (T0 - T1) / ice.constants.Lf
from scipy.optimize import root
from scipy.special import erf

lam = root(lambda x: x * np.exp(x ** 2) * erf(x) - Stf / np.sqrt(np.pi), 1)['x'][0]
time_length = ((0.9 * Lz) / (2 * lam)) ** 2 / kappa

nt = int(time_length / dt)
print(time_length, nt)

ice.stefan_compare()
ice.stefan_solution(nt * dt, T1=Ttop, T0=Tbot)
time = np.asarray(ice._time_)
norm_time = time / (max(time) - min(time))
model_front = np.asarray(ice.freeze_front)

plt.figure(1)

plt.plot(norm_time, ice.stefan_zm_funct(time), color='k', linewidth=3)
plt.plot(norm_time, model_front)
plt.xlabel('normalized time')
plt.ylabel('freezing front position')
plt.show()

# multiple runs
'''
# numerical parameters
Lz = 10e3
dz = 10
dt = 3.14e7 / 6

# physical parameters
Ttop = 110
Tbot = Tsill = 273.15
thicks = [0.5e3, 0.75e3, 1e3, 1.25e3, 1.5e3, 1.75e3, 2e3]
depth = 1e3

# solution arrays
model_solution = np.zeros(len(thicks))
stefan_solution = model_solution.copy()


for i, thick in enumerate(thicks):
	ice = IceSystem(Lz = Lz, dz= dz, dt=dt)
	ice.init_T(Ttop, Tbot)
	T1 = ice.T[np.where((ice.z >= depth-dz) & (ice.z<= depth+dz))]
	T0 = Tbot
	ice.init_sill(Tsill=Tbot, depth=depth, thickness=thick, cpT='non')
	ice.set_BC()
	ice.freezestop = 1
	ice.solve_heat(int(1e13))
	model_solution[i] = ice.freeze_time
	ice.stefan_solution(ice.freeze_time, T1=T1[0], T0=T0)
	stefan_solution[i] = ice.stefan_time_frozen

plt.plot(thicks, stefan_solution, label='stefan time')
plt.plot(thicks, model_solution, label='model time')
plt.legend()
plt.show()
'''

# Test sill freeze, MM style
# don't know what to do with this
'''
Ttop = 110
Tbot = 272
ice = IceSystem(Lz, dz, dt)
ice.init_T(Ttop, Tbot)
depth = 1e3
ice.constants.rho_i = 910.
ice.constants.kw = 0.6
#ice.init_vol_avgs(cpT='non')
ice.init_sill(Tsill=273.15, depth=depth, thickness=0.5e3, phi=1, cpT='non')
#ice.T[1:] = Tbot
#ice.phi[0] = 1
T_i, phi_i, k_i, rhoc_i = ice.T.copy(), ice.phi.copy(), ice.k.copy(), ice.rhoc.copy()
ice.freezestop = 1
nt = int(1e12/dt)

ft = ice.MM_solver(nt)
#ice.stefan_solution(nt*dt,T1=Ttop, T0=Tbot)


# plot results
plt.figure(1)
plt.gca().invert_yaxis()
plt.plot(T_i, ice.z,label='initial')
plt.plot(ice.T, ice.z, label='t = {:0.03f}yr'.format(nt*dt/ice.constants.styr))
#plt.plot(ice.stefan_T, ice.stefan_z, label='stefan soln' )
#plt.plot([Ttop, Tbot], [ice.stefan_zm]*2, 'k--', label='stefan melting front')
plt.legend()
plt.xlabel('Temperature (K)')
plt.ylabel('depth (k)')


plt.figure(2)
plt.gca().invert_yaxis()
plt.plot(phi_i, ice.z,label='initial')
plt.plot(ice.phi, ice.z, label='t = {:0.03f}yr'.format(nt*dt/ice.constants.styr))
#plt.plot([0,1], [ice.stefan_zm]*2, 'k--',label='stefan melting front')
plt.xlabel('liquid fraction')
plt.ylabel('depth (m)')
plt.legend()
plt.show()
'''
