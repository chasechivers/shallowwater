from modular_build import IceSystem
import matplotlib.pyplot as plt
import numpy as np

Lz = 1.
Lx = 0.015
dx = dz = 0.005
dt = 0.1

Tsurf = 50.
Tbot = 273.15

ice = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, cpT=False, kT=False)
ice.init_T(Tsurf=Tsurf, Tbot=Tbot)
ice.T[1:, :] = 273.15
ice.phi[:, :] = 1
ice.init_volume_averages()
ice.set_boundayconditions(sides='Reflect')
ice.stefan_compare(dt)

kappa = ice.constants.ki / (ice.constants.cp_i * ice.constants.rho_i)
Stf = ice.constants.cp_i * (Tbot - Tsurf) / ice.constants.Lf
from scipy.optimize import root, curve_fit
from scipy.special import erf

lam = root(lambda x: x * np.exp(x ** 2) * erf(x) - Stf / np.sqrt(np.pi), 1)['x'][0]
time_length = ((0.9 * Lz) / (2 * lam)) ** 2 / kappa

nt = int(time_length / dt)
print(time_length, nt)

time = np.asarray(ice._time_)
norm_time = time / (max(time) - min(time))
model_front = np.asarray(ice.freeze_front)

plt.figure(1)

plt.plot(norm_time, ice.stefan_zm_func(time), color='k', linewidth=3)
plt.plot(norm_time, model_front)
plt.xlabel('normalized time')
plt.ylabel('freezing front position')
plt.show()

plt.figure(2)
plt.plot(norm_time, model_front - ice.stefan_zm_func(time))
plt.xlabel('normalized time')
plt.ylabel('error in freezing front position')
plt.show()

from scipy.optimize import root, curve_fit

print(curve_fit(lambda x, C: C * np.sqrt(x), time, model_front)[0] / ice.stefan_zm_const)
