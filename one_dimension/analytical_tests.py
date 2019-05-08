from one_dimension.modular_1D import IceSystem
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette='colorblind', style='whitegrid')
# Test Stefan Problem

Lz = 1
dz = 0.005
dt = 2.5
ice = IceSystem(Lz, dz)
Ttop = T1 = 110
Tbot = T0 = 273.15
ice.init_T(Ttop, Tbot)
ice.T[1:] = 273.15
ice.phi[:] = 1.
# ice.constants.rho_w = ice.constants.rho_i
# ice.constants.cp_w = ice.constants.cp_i
# ice.constants.kw = ice.constants.ki
ice.init_vol_avgs(kT='non', cpT='non')
T_i, phi_i, k_i = ice.T.copy(), ice.phi.copy(), ice.k.copy()
ice.set_BC()

# determine time needed to freeze whole domain
kappa = ice.constants.ki / (ice.constants.cp_i * ice.constants.rho_i)
Stf = ice.constants.cp_i * (T0 - T1) / ice.constants.Lf
from scipy.optimize import root, curve_fit
from scipy.special import erf

lam = root(lambda x: x * np.exp(x ** 2) * erf(x) - Stf / np.sqrt(np.pi), 1)['x'][0]
time_length = ((0.9 * Lz) / (2 * lam)) ** 2 / kappa

nt = int(time_length / dt)
print(time_length, nt)

ice.stefan_compare(dt=dt, OF=1)
ice.stefan_solution(nt * dt, T1=Ttop, T0=Tbot)

time = np.asarray(ice._time_)
model_front = np.asarray(ice.freeze_front)

C_fit, _ = curve_fit(lambda x, C: C * np.sqrt(x), time, model_front)

norm_time = time / (max(time) - min(time))

plt.figure(1)
plt.plot(norm_time, ice.stefan_zm_func(time), color='k', linewidth=4, label='analytical')
plt.plot(norm_time, model_front, label='model')
plt.plot(norm_time, C_fit * np.sqrt(time), label='model fit')
plt.xlabel('normalized time')
plt.ylabel('freezing front position')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(norm_time, model_front - ice.stefan_zm_func(time), label='model')
plt.plot(norm_time, C_fit * np.sqrt(time) - ice.stefan_zm_func(time), label='fit')
plt.xlabel('normalized time')
plt.ylabel('error in freezing front position')
plt.legend()
plt.show()
'''

# Sweep some numerical parameters to find bests
# dz equivalence for Lz = 10e3
#    100 m, 50 m, 25 m, 10 m
dzs = [0.01, 0.005, 0.0025, 0.001]

dts = [5, 2.5, 1, 0.5, 0.1, 0.05]

C_fits = np.zeros((len(dzs), len(dts)))
Ttop = T1 = 110
Tbot = T0 = 273.15
Lz = 1

ice = IceSystem(Lz, 0.1)
ice.init_T(Ttop, Tbot)

# determine how long to run simulation
# and time length for 90% of domain to freeze
kappa = ice.constants.ki / (ice.constants.cp_i * ice.constants.rho_i)
Stf = ice.constants.cp_i * (T0 - T1) / ice.constants.Lf
from scipy.optimize import root, curve_fit
from scipy.special import erf

lam = root(lambda x: x * np.exp(x ** 2) * erf(x) - Stf / np.sqrt(np.pi), 1)['x'][0]
time_length = ((0.9 * Lz) / (2 * lam)) ** 2 / kappa
ice.stefan_solution(time_length, T1=Ttop, T0=Tbot)
stf_zm = ice.stefan_zm_func
stf_C = ice.stefan_zm_const

# tests for constant and equal thermal properties
for i, dz in enumerate(dzs):
	for j, dt in enumerate(dts):
		ice = IceSystem(Lz, dz)
		ice.init_T(Ttop, Tbot)
		ice.T[1:] = 273.15
		ice.phi[:] = 1.
		ice.constants.rho_w = ice.constants.rho_i
		ice.constants.cp_w = ice.constants.cp_i
		ice.constants.kw = ice.constants.ki
		ice.init_vol_avgs(kT='non', cpT='non')
		ice.set_BC()
		nt = int(time_length / dt)
		print(dz, dt, nt)
		ice.stefan_compare(dt=dt, OF=1)
		model_front = np.asarray(ice.freeze_front)
		time = np.asarray(ice._time_)
		C_fits[i, j], _ = curve_fit(lambda x, C: C * np.sqrt(x), time, model_front)
		norm_time = time / (max(time) - min(time))
		plt.figure(1)
		plt.plot(norm_time, C_fits[i, j] * np.sqrt(time) - stf_zm(time), label='dt={}, dz={}'.format(dt, dz))
		plt.xlabel('normalized time')
		plt.ylabel('error in freezing front position')
		plt.xlim(0,1)
		plt.legend()
		plt.pause(0.001)

ice.stefan_solution(time_length, T1=Ttop, T0=Tbot)
'''
